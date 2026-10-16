import os
import json
import time
import re
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Google Cloud Vision API
from google.cloud import vision
from google.oauth2 import service_account

# Google GenAI SDK (Gemini)
from google import genai
from google.genai import types

# Pydantic V1 / V2 互換性設定
try:
    from pydantic import ConfigDict
    HAS_PYDANTIC_V2 = True
except ImportError:
    HAS_PYDANTIC_V2 = False

# Supabase DATABASE_URL の接続文字列変換
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

Base = declarative_base()

class ExpenseDB(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, index=True)
    amount = Column(Float)
    category = Column(String)
    memo = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

app = FastAPI()

# 1. Vision API クライアントの初期化（環境変数 GOOGLE_APPLICATION_CREDENTIALS_JSON）
vision_client = None
gcp_json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if gcp_json_str:
    try:
        credentials_info = json.loads(gcp_json_str)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        print(f"Vision API Credentials Error: {e}")

# 2. Gemini クライアントの初期化（環境変数 GEMINI_API_KEY）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.on_event("startup")
def startup_event():
    if engine:
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            print(f"DB Init Warning: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    if not SessionLocal:
        raise HTTPException(status_code=500, detail="Database URL not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ExpenseCreate(BaseModel):
    date: str
    amount: float
    category: str
    memo: Optional[str] = None

class ExpenseResponse(BaseModel):
    id: int
    date: str
    amount: float
    category: str
    memo: Optional[str] = None

    if HAS_PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:
        class Config:
            orm_mode = True

class ReceiptParseResult(BaseModel):
    amount: Optional[float] = None
    date: Optional[str] = None
    category: Optional[str] = "食費"
    memo: Optional[str] = None

@app.get("/api/expenses", response_model=List[ExpenseResponse])
def read_expenses(db: Session = Depends(get_db)):
    return db.query(ExpenseDB).order_by(ExpenseDB.id.desc()).all()

@app.post("/api/expenses", response_model=ExpenseResponse)
def create_expense(expense: ExpenseCreate, db: Session = Depends(get_db)):
    db_expense = ExpenseDB(
        date=expense.date,
        amount=expense.amount,
        category=expense.category,
        memo=expense.memo
    )
    db.add(db_expense)
    db.commit()
    db.refresh(db_expense)
    return db_expense

@app.delete("/api/expenses/{expense_id}")
def delete_expense(expense_id: int, db: Session = Depends(get_db)):
    expense = db.query(ExpenseDB).filter(ExpenseDB.id == expense_id).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    db.delete(expense)
    db.commit()
    return {"message": "Deleted successfully"}

# ハイブリッド レシート解析API（Vision API [OCR] + Gemini 3.8 Flash [Text-to-JSON]）
@app.post("/api/scan-receipt")
async def scan_receipt(file: UploadFile = File(...)):
    t_start = time.time()
    
    if not vision_client:
        raise HTTPException(status_code=500, detail="GOOGLE_APPLICATION_CREDENTIALS_JSON is not configured")
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    try:
        image_bytes = await file.read()
        file_size_kb = len(image_bytes) / 1024
        print(f"⏱️ [ログ] 画像受信完了: {file_size_kb:.1f} KB")

        # --- STEP 1: Vision API で文字起こし ---
        t_vision_start = time.time()
        image = vision.Image(content=image_bytes)
        vision_response = vision_client.text_detection(image=image)

        if vision_response.error.message:
            raise Exception(f"Vision API Error: {vision_response.error.message}")

        full_text = vision_response.full_text_annotation.text if vision_response.full_text_annotation else ""
        t_vision_end = time.time()
        print(f"⏱️ [ログ] 1/2 Vision API OCR時間: {t_vision_end - t_vision_start:.2f}秒")

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="レシートから文字を検出できませんでした")

        # --- STEP 2: Gemini でテキスト構造化 (JSON化) ---
        t_gemini_start = time.time()
        prompt = f"""以下はレシートから抽出された生のテキストです。
ここから amount(数値), date(YYYY/MM/DD形式), category(食費/日用品/交通費/交際費/その他), memo(店舗名や概要) を抽出してJSONで返してください。

【レシートテキスト】:
{full_text}"""

        config = types.GenerateContentConfig(
            system_instruction="You are an expert OCR parser. Return strictly valid JSON adhering to the schema.",
            response_mime_type="application/json",
            response_schema=ReceiptParseResult,
            temperature=0.0,
        )

        response = await gemini_client.aio.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=config,
        )
        t_gemini_end = time.time()
        print(f"⏱️ [ログ] 2/2 Gemini JSON生成時間: {t_gemini_end - t_gemini_start:.2f}秒")
        print(f"⚡ [ログ] 全体処理時間: {t_gemini_end - t_start:.2f}秒")

        # JSON結果抽出
        result = None
        if hasattr(response, 'parsed') and response.parsed:
            if hasattr(response.parsed, 'model_dump'):
                result = response.parsed.model_dump()
            elif hasattr(response.parsed, 'dict'):
                result = response.parsed.dict()
            else:
                result = response.parsed

        if not result and getattr(response, 'text', None):
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

        if not result:
            raise ValueError("AIからの応答データからJSONを抽出できませんでした")

        return result

    except Exception as e:
        print(f"Hybrid Scan Error: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def read_index():
    return FileResponse("public/index.html")