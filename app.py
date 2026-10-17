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
        image = Image.open(io.BytesIO(file.read()))

        # AI処理の最速設定
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            system_instruction="Extract date, store_name, total_amount from receipt image."
        )

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[image, "Return JSON only."],
            config=config
        )

        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)