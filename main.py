import os
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import google.generativeai as genai

app = FastAPI(title="Receipt OCR API")

# CORSの設定（フロントエンドからのリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 1. クライアントのグローバル初期化（SSL/接続オーバーヘッドを排除）
# -------------------------------------------------------------
# リクエストごとに作成せず、アプリ起動時に1度だけ生成して使い回す
vision_client = vision.ImageAnnotatorClient()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing")

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------------------------------------
# 2. 超高速レスポンス設定（generation_config）
# -------------------------------------------------------------
# - response_mime_type="application/json": マークダウン(```json)や雑談メッセージの生成を完全に阻止
# - temperature=0.0: 推論の揺らぎをゼロにして最速で回答
# - max_output_tokens=400: 余計な長文出力をカット
generation_config = {
    "response_mime_type": "application/json",
    "temperature": 0.0,
    "max_output_tokens": 400,
}

# 爆速・軽量モデル指定（有料枠のためレート制限を気にせず使えます）
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# -------------------------------------------------------------
# 3. プロンプトの極小化（入力トークン処理時間を最小化）
# -------------------------------------------------------------
SYSTEM_PROMPT = """Extract receipt info from OCR text into strictly formatted JSON.
Fields:
- store_name (string)
- date (YYYY-MM-DD or string)
- total_amount (number or null)
- items (array of {name: string, price: number})

OCR Text:
"""

@app.get("/")
async def root():
    return {"status": "ok", "message": "Receipt OCR API is running"}

@app.post("/api/scan-receipt")
async def scan_receipt(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # 1. 画像データの読み込み
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided")
            
        # 2. Vision API で高速テキスト抽出 (OCR)
        t_vision_start = time.time()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            raise HTTPException(status_code=500, detail=f"Vision API error: {response.error.message}")
            
        texts = response.text_annotations
        if not texts:
            return {
                "success": False,
                "message": "No text detected in receipt",
                "data": None
            }
            
        ocr_text = texts[0].description
        vision_duration = time.time() - t_vision_start

        # 3. Gemini で JSON 構造化・整列（高速処理）
        t_gemini_start = time.time()
        full_prompt = SYSTEM_PROMPT + ocr_text
        
        gemini_response = gemini_model.generate_content(full_prompt)
        gemini_duration = time.time() - t_gemini_start
        
        # 4. JSONパースと計測時間の返却
        parsed_data = json.loads(gemini_response.text)
        total_duration = time.time() - start_time
        
        return {
            "success": True,
            "data": parsed_data,
            "performance": {
                "vision_api_sec": round(vision_duration, 3),
                "gemini_api_sec": round(gemini_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse Gemini output as JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))