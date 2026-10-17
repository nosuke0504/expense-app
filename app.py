import os
import io
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from google import genai

app = Flask(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        return "ファイルが送信されていません", 400
    
    file = request.files['file']
    if file.filename == '':
        return "ファイルが選択されていません", 400
    
    try:
        # 1. 画像処理：800pxまで小さくし、JPEG圧縮で通信量を激減させる
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image.thumbnail((800, 800))
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        compressed_image = Image.open(buffer)

        # 2. AI処理：必要な項目だけを即答させるプロンプト
        prompt = "レシート画像から「日付」「店舗名」「合計金額」を読み取り、JSON形式のみで出力してください。"
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[compressed_image, prompt]
        )

        return f"<h3>解析結果:</h3><pre>{response.text}</pre><br><a href='/'>再度撮影する</a>"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)