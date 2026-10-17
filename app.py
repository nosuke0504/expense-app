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
    # ページを直接リロードした場合はトップへ戻す（405エラー防止）
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        return "ファイルが送信されていません", 400
    
    file = request.files['file']
    if file.filename == '':
        return "ファイルが選択されていません", 400
    
    try:
        # 画像を読み込み、縦横最大1024pxに縮小して処理を高速化
        image = Image.open(io.BytesIO(file.read()))
        image.thumbnail((1024, 1024))

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