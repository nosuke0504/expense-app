from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import random

app = FastAPI()

# public フォルダを静的ファイル（HTML/CSS/JS）として公開
app.mount("/static", StaticFiles(directory="public"), name="static")


@app.get("/")
def read_root():
    # トップアクセス時に public/index.html を返す
    with open("public/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# 1. カメラ画像を受け取ってリアルタイム照合するAPI（かざすだけスキャナー用）
@app.post("/api/scan")
async def scan_receipt(file: UploadFile = File(...)):
    # 受信した画像データを読み込み（※ここで将来的にAI OCRや画像解析を実行）
    image_bytes = await file.read()
    image_size = len(image_bytes)

    # 【ダミー照合ロジック】
    # 実際はここで画像から日付・金額を判定し、DBと照合します
    # ここではテスト用にランダムで確定Noを割り当てるデモ動作をします
    matched = random.choice([True, False, False])  # 3回に1回マッチするテスト用

    if matched:
        confirmed_no = random.randint(1, 15)
        return {
            "found": True,
            "confirmed_no": confirmed_no,
            "message": f"確定 No. {confirmed_no} と一致しました！",
        }
    else:
        return {
            "found": False,
            "message": "照合中...",
        }


# 2. データ手動登録時のルール事前検証API
@app.post("/api/validate")
async def validate_receipt(
    date: str = Form(...),
    amount: int = Form(...),
    has_stamp: bool = Form(False),
    recipient: str = Form(""),
):
    warnings = []

    # ① 対象期間外チェック (2026/04/01 ～ 2027/03/31)[cite: 1]
    if not (date >= "2026-04-01" and date <= "2027-03-31"):
        warnings.append("【警告】支出日付が補助費の対象期間（2026/4/1～2027/3/31）外です。")

    # ② 5万円以上・現金時の収入印紙チェック[cite: 1]
    if amount >= 50000 and not has_stamp:
        warnings.append("【警告】5万円以上の支払いです。収入印紙の貼付・割印、または不貼付理由の記載が必要です。")

    # ③ 宛名チェック[cite: 1]
    if not recipient or "部" not in recipient:
        warnings.append("【注意】宛名に「東邦大学 ○○部」のように大学名と団体名が正しく記載されているか確認してください。")

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)