import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# ==========================================
# 1. Google Cloud / Vision API 用の認証設定
# ==========================================
# Renderの環境変数からJSON文字列を取得して認証ファイルを作成
google_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if google_creds_json:
    creds_path = "/tmp/google-credentials.json"
    with open(creds_path, "w") as f:
        f.write(google_creds_json)
    # Googleライブラリが参照する環境変数をセット
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# 動作確認（Vision APIを使用する場合）
try:
    from google.cloud import vision

    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    vision_client = None

# ==========================================
# 2. Gemini API の設定
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 3. FastAPI アプリケーション本体
# ==========================================
app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    """動作確認用ルート"""
    return {"status": "ok", "message": "Server is running"}


@app.post("/generate")
def generate_text(request: ChatRequest):
    """Gemini APIを使用してテキストを生成するエンドポイント"""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY が設定されていません。"
        )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(request.prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))