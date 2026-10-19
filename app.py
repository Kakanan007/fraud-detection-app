import gradio as gr
import pandas as pd
import joblib

# 1. โหลดโมเดลและคอลัมน์
model = joblib.load('fraud_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. ฟังก์ชันทำนายความเสี่ยง
def predict_fraud(amount, tx_freq_1h, distance_km, channel, merchant_type):
    input_data = pd.DataFrame([{
        'amount': amount,
        'tx_freq_1h': tx_freq_1h,
        'distance_km': distance_km,
        'channel': channel,
        'merchant_type': merchant_type
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(input_encoded)[0][1]
    
    labels = {
        "โอกาสเป็นปกติ (Normal)": float(1 - prob),
        "โอกาสทุจริต (Fraud)": float(prob)
    }

    if prob >= 0.7:
        action = "🚨 ความเสี่ยงสูง: อายัดบัตรชั่วคราว และส่งเรื่องให้ทีม Fraud ตรวจสอบทันที"
    elif prob >= 0.3:
        action = "⚠️ ความเสี่ยงปานกลาง: ส่ง SMS ขอรหัส OTP ยืนยันตัวตนผู้ถือบัตร"
    else:
        action = "✅ ความเสี่ยงต่ำ: อนุมัติทำรายการสำเร็จ"

    return labels, action

# 3. สร้าง UI ด้วย Gradio Blocks ธีม Monochrome (ขาว-ดำ-เทา เรียบหรู)
custom_theme = gr.themes.Monochrome(
    primary_hue="neutral",
    secondary_hue="neutral",
)

with gr.Blocks(theme=custom_theme) as demo:
    # ส่วนหัวข้อหลัก
    with gr.Row():
        gr.Markdown(
            """
            # 💳 ระบบตรวจจับการทุจริตธุรกรรมบัตรเครดิต
            *ระบบจำลองการประเมินความเสี่ยงและตัดสินใจอัตโนมัติ (Automated Fraud Detection)*  
            กรอกข้อมูลรายละเอียดการทำรายการด้านล่างเพื่อให้ระบบวิเคราะห์ความผิดปกติแบบ Real-time
            """
        )

    with gr.Row(equal_height=True):
        # คอลัมน์ซ้าย: สำหรับกรอกข้อมูล (Input Form)
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📝 ข้อมูลการทำรายการ (Transaction Details)")
            amount_input = gr.Number(label="ยอดเงินที่ใช้ (บาท)", value=1500.0)
            freq_input = gr.Slider(label="ความถี่การใช้บัตรใน 1 ชม. (ครั้ง)", minimum=0, maximum=10, value=1, step=1)
            dist_input = gr.Number(label="ระยะห่างจากที่อยู่ประจำ (กม.)", value=5.0)
            channel_input = gr.Dropdown(label="ช่องทางชำระเงิน", choices=["Online", "POS_Swipe", "Contactless"], value="Online")
            merchant_input = gr.Dropdown(label="ประเภทร้านค้า", choices=["Dining", "Digital_Goods", "Luxury", "Travel", "Retail"], value="Dining")
            
            btn_predict = gr.Button("🚀 ประเมินความเสี่ยง", variant="primary", size="lg")

        # คอลัมน์ขวา: สำหรับแสดงผลลัพธ์ (Output Results)
        with gr.Column(scale=1, variant="compact"):
            gr.Markdown("### 📊 ผลการวิเคราะห์จากระบบ (AI Evaluation)")
            output_label = gr.Label(label="ระดับความน่าจะเป็น (Probability)")
            output_action = gr.Textbox(label="คำสั่งระบบอัตโนมัติ (Action Automation)", lines=3)

    # ผูกปุ่มกดเข้ากับฟังก์ชัน
    btn_predict.click(
        fn=predict_fraud,
        inputs=[amount_input, freq_input, dist_input, channel_input, merchant_input],
        outputs=[output_label, output_action]
    )

# 4. รันสำหรับเซิร์ฟเวอร์ Render
demo.launch(server_name="0.0.0.0", server_port=10000)
