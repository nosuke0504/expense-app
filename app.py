import os
import io
import json
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
        return jsonify({"success": False, "error": "ファイルが送信されていません"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "ファイルが選択されていません"}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=60)
        
        image_part = types.Part.from_bytes(
            data=img_byte_arr.getvalue(),
            mime_type="image/jpeg"
        )

        # JSONレスポンスの構造を厳密に指定
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            system_instruction=(
                "Extract date (YYYY-MM-DD), store_name, and total_amount (number only) from receipt. "
                "Return JSON with keys: 'date', 'store_name', 'total_amount'."
            )
        )

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[image_part, "Extract receipt info."],
            config=config
        )

        # Geminiから返ってきたJSON文字列をパース
        parsed_data = json.loads(response.text)

        return jsonify({"success": True, "data": parsed_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/save', methods=['POST'])
def save():
    try:
        data = request.json
        print("保存されたデータ:", data)
        # ※ ここにスプレッドシートやDBへの書き込み処理を接続できます
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)