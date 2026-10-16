import os
import json
import time
import re
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Gemini SDK
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

# Gemini クライアントの初期化（サーバー起動時に1度だけ生成）
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

# Gemini AI レシート解析API（思考スキップ＆爆速即答化設定）
@app.post("/api/scan-receipt")
async def scan_receipt(file: UploadFile = File(...)):
    t_start = time.time()
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set in environment variables")

    try:
        image_bytes = await file.read()
        t_upload = time.time()
        file_size_kb = len(image_bytes) / 1024
        print(f"⏱️ [ログ] 画像受信完了: {file_size_kb:.1f} KB (受信時間: {t_upload - t_start:.2f}秒)")

        prompt = (
            "レシート画像から以下を抽出してJSON出力:\n"
            "amount: 支払合計金額(数値のみ)\n"
            "date: 日付(YYYY-MM-DD)\n"
            "category: 食費,日用品,交通費,娯楽,その他のいずれか\n"
            "memo: 店舗名"
        )

        t_ai_start = time.time()

        # 思考（Thinking）処理を完全に0にし、必要最小限のトークン数で即答させる
        try:
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            thinking_config = None

        config_args = {
            "response_mime_type": "application/json",
            "response_schema": ReceiptParseResult,
            "max_output_tokens": 100,
            "temperature": 0.1,
        }
        if thinking_config:
            config_args["thinking_config"] = thinking_config

        response = gemini_client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type or "image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(**config_args),
        )

        t_ai_end = time.time()
        print(f"⏱️ [ログ] Gemini AI応答時間: {t_ai_end - t_ai_start:.2f}秒 (全体時間: {t_ai_end - t_start:.2f}秒)")

        # 解析結果の安全な抽出
        result = None
        if hasattr(response, 'parsed') and response.parsed:
            if hasattr(response.parsed, 'model_dump'):
                result = response.parsed.model_dump()
            elif hasattr(response.parsed, 'dict'):
                result = response.parsed.dict()
            else:
                result = response.parsed

        if not result and response.text:
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_text)

        if not result:
            raise ValueError("AIからの応答データを解析できませんでした")

        return result

    except Exception as e:
        print(f"Gemini OCR Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {str(e)}")

app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def read_index():
    return FileResponse("public/index.html")