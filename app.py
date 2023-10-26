from flask import Flask, render_template
from ultralytics import YOLO

app = Flask(__name__)

API_URL = 0

@app.route('/')
def home():
    return "Selamat datang di Aplikasi Deteksi Pelanggaran"

@app.route('/detect')
def detect_objects():
    try:
        model = YOLO('model/best.pt')
        result = model(source=API_URL, conf=0.6)
        object_count = len(result.pred[0])
        return render_template('pages/hasil_deteksi.html', object_count= object_count, api_url= API_URL)
    except Exception as e:
        error_message = f"Terjadi kesalahan saat mengakses kamera CCTV: {str(e)}"
        return render_template('pages/error.html', error_message = error_message)
    
if __name__ == '__main__':
    app.run()