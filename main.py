import os
import json
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

# AI解析レスポンス構造体
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

# Gemini AI レシート解析API
@app.post("/api/scan-receipt")
async def scan_receipt(file: UploadFile = File(...)):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set in environment variables")

    try:
        image_bytes = await file.read()
        client = genai.Client(api_key=api_key)

        prompt = (
            "提供されたレシート画像から以下の情報を抽出し、JSONで出力してください。\n"
            "1. amount: 支払合計金額（数値のみ。例: 1280）\n"
            "2. date: 購入日付（YYYY-MM-DD形式。例: 2026-10-17。年が不明な場合は今年とする）\n"
            "3. category: カテゴリ（'食費', '日用品', '交通費', '娯楽', 'その他' のいずれか1つを判定）\n"
            "4. memo: 店舗名や主な購入物の概要（例: サイゼリヤ）\n"
        )

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type or "image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReceiptParseResult,
            ),
        )

        result = json.loads(response.text)
        return result

    except Exception as e:
        print(f"Gemini OCR Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {str(e)}")

app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def read_index():
    return FileResponse("public/index.html")