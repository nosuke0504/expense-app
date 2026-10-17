import os
import io
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from google import genai
from google.genai import types
import openpyxl
from openpyxl.styles import PatternFill

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
            "- doc_type: 書類種別（「領収書」、「請求書」、「納品書」、「注文書」、「受領書」など）\n"
            "- warnings: ガイドラインに基づく不備警告メッセージの配列。以下のルールを評価してください：\n"
            "  1. 宛名が「東邦大学〇〇部」形式でない場合: '宛名が大学・団体名ではありません（個人名・立替の場合は部員名簿等の添付が必要です）。'\n"
            "  2. 但書が「品代」や「備品代」など具体的でない場合: '但書が曖昧です（具体的な購入品がわかる明細レシート等を添付してください）。'\n"
            "  3. 金額が50,000円以上の場合: '5万円以上の支出です（収入印紙・割印の貼付、および使途・数量等の説明資料が必要です）。'\n"
            "  4. doc_typeが「領収書」「領収証」「領収証書」以外（請求書、納品書、注文書、受領書など）の場合: '「領収証」以外の書類は不可です（支払事実が確認できる代用資料または領収証が必要です）。'\n"
            "  5. 日付が該当年度（4月1日〜3月31日）の範囲外の場合: '補助費の対象年度（4/1〜3/31）外の支出の可能性があります。'\n"
            "  6. 大会エントリー費・分担金・参加費などの場合: '大会の概要（費用・主催者・支払先が判る資料）の添付が必要です。'\n"
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

@app.route('/export_excel', methods=['POST'])
def export_excel():
    try:
        data = request.get_json() or {}
        receipts = data.get('receipts', [])
        subsidy = data.get('subsidy', 0)
        group_name = data.get('group_name', '○○部')
        author_name = data.get('author_name', '')

        template_filename = '学生団体（各部活動）202506.xlsx'
        if not os.path.exists(template_filename):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = '決算報告'
        else:
            wb = openpyxl.load_workbook(template_filename)
            ws = wb['決算報告'] if '決算報告' in wb.sheetnames else wb.active

        # 背景色を白一色に設定するためのスタイル
        white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

        # 1. ヘッダー基本情報の書き込み
        ws.cell(row=2, column=5, value=group_name)  # 団体名 (E2)
        ws.cell(row=3, column=5, value=author_name) # 作成者名 (E3)
        ws.cell(row=7, column=5, value=subsidy)     # 補助支給額 (E7)

        # 2. 支出明細の書き込み (14行目から開始)
        start_row = 14
        for i, item in enumerate(receipts):
            r = start_row + i
            amt = item.get('total_amount', 0)
            try:
                amt = float(amt)
            except (ValueError, TypeError):
                amt = 0

            ws.cell(row=r, column=2, value=i + 1)                      # 通し番号 (B列)
            ws.cell(row=r, column=3, value=item.get('date', ''))        # 支払年月日 (C列)
            ws.cell(row=r, column=4, value=item.get('proviso', ''))     # 摘要/但書 (D列)
            ws.cell(row=r, column=5, value=amt)                        # 支出金額 (E列)
            ws.cell(row=r, column=6, value=item.get('store_name', ''))  # 備考 (F列)

        # 3. 全セルの背景色を白一色に変更
        for row in ws.iter_rows(min_row=1, max_row=max(100, start_row + len(receipts) + 5), min_col=1, max_col=10):
            for cell in row:
                cell.fill = white_fill

        output = io.BytesIO()
        wb.save(output)
        output.seek(0)

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='東邦大学青藍会_決算報告書.xlsx'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)