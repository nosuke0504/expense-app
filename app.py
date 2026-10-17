import os
import io
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from google import genai
from google.genai import types

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
        image = Image.open(io.BytesIO(file.read()))

        # AIの思考・生成速度を最速にする設定
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )

        prompt = "レシートから「日付」「店舗名」「合計金額」を読み取り、JSON形式のみで出力してください。"
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[image, prompt],
            config=config
        )

        return f"<h3>解析結果:</h3><pre>{response.text}</pre><br><a href='/'>再度撮影する</a>"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)