import os
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "ファイルが送信されていません"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    try:
        # 画像データを読み込み・軽量JPEGへ変換
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=60)
        
        # API送信用のPartオブジェクトを高速作成
        image_part = types.Part.from_bytes(
            data=img_byte_arr.getvalue(),
            mime_type="image/jpeg"
        )

        system_instruction = (
            "あなたは東邦大学青藍会補助費決算報告用の領収証・レシート画像解析システムです。"
            "画像から以下の項目を抽出し、JSONフォーマットで返却してください。\n"
            "- date: 発行日または注文日 (YYYY-MM-DD形式)\n"
            "- store_name: 支払先・店舗名\n"
            "- addressee: 宛名（「東邦大学〇〇部」、「個人名」、「記載なし」など）\n"
            "- proviso: 但書（例: 「○○購入代」、「品代」など）\n"
            "- total_amount: 合計金額（数値のみ）\n"
            "- payment_method: 支払方法（「現金」、「クレジットカード」、「不明」など）\n"
            "- warnings: ガイドラインに基づく不備警告メッセージの配列。以下のルールを評価してください：\n"
            "  1. 宛名が「東邦大学〇〇部」形式でない場合（なし、または個人名）: '宛名が大学・団体名ではありません（個人名の場合は部員名簿等の添付が必要です）。'\n"
            "  2. 但書が「品代」や「備品代」など具体的でない場合: '但書が曖昧です（具体的な購入品がわかる明細レシート等を添付してください）。'\n"
            "  3. 金額が50,000円以上の場合: '5万円以上の支出です（収入印紙・割印の貼付、またはクレジットカード利用の明記が必要です）。'\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            system_instruction=system_instruction
        )

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[image_part, "Extract details as JSON according to system instruction."],
            config=config
        )

        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)