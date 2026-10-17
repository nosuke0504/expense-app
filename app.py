import os
import io
from flask import Flask, render_template, request
from PIL import Image
from google import genai

app = Flask(__name__)

# Renderの環境変数からGemini APIキーを取得
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "ファイルが送信されていません", 400
    
    file = request.files['file']
    
    try:
        # 画像データを読み込み
        image = Image.open(io.BytesIO(file.read()))

        # 最新モデル gemini-3.8-flash を指定
        prompt = "このレシート画像から「日付」「店舗名」「合計金額」を正確に抽出し、見やすいJSON形式で回答してください。"
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[image, prompt]
        )

        return f"<h3>解析結果:</h3><pre>{response.text}</pre><br><a href='/'>再度撮影する</a>"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)