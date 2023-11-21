from flask import Flask, render_template, session, Response
from YoloVideo import video_detection
import cv2

app = Flask(__name__)

app.config['SECRET_KEY'] = 'taufiq'

import cv2

def generate_frame(path):
    for frame in video_detection(path):
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route("/", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/webapp')
def webapp():
    return Response(generate_frame(path=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)