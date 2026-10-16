import os
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Pydantic V1 / V2 両対応の互換性設定
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

# テーブル定義
class ExpenseDB(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, index=True)
    amount = Column(Float)
    category = Column(String)
    memo = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

app = FastAPI()

# 起動時に安全にテーブルを作成
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

# スキーマ定義
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

# API エンドポイント
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

# 静的ファイルの配信
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def read_index():
    return FileResponse("public/index.html")