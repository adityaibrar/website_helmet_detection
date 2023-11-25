from flask import Flask, render_template, Response
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone

app = Flask(__name__)

app.config['SECRET_KEY'] = 'taufiq'

def draw_rectangle_and_text(frame, bbox, text, color):
    x, y, w, h, id1 = bbox
    cxm, cym = (x + w) // 2, (y + h) // 2

    cv2.circle(frame, (cxm, cym), 4, color, -1)
    cv2.rectangle(frame, (x, y), (w, h), color, 1)
    cvzone.putTextRect(frame, f'{text} {id1}', (x, y), 1, 1)

def objectcount(frame, model, class_list, tracker1, tracker2, tracker3, counter_raider, counter_nohelmet, counter_helmet):
    results = model(source=frame, conf=0.5)
    bboxes = pd.DataFrame(results[0].boxes.data).astype("int")

    raider_bboxes = []
    nohelmet_bboxes = []
    helmet_bboxes = []

    offset = 6
    
    cy1 = 424

    for _, row in bboxes.iterrows():
        x1, y1, x2, y2, _, d = row
        class_name = class_list[d]
        if 'raider' in class_name:
            draw_rectangle_and_text(frame, [x1, y1, x2, y2, d], class_name, (255, 0, 0))
        elif 'nohelmet' in class_name:
            draw_rectangle_and_text(frame, [x1, y1, x2, y2, d], class_name, (0, 0, 255))
        elif 'helmet' in class_name:
            draw_rectangle_and_text(frame, [x1, y1, x2, y2, d], class_name, (0, 255, 0))

    for _, row in bboxes.iterrows():
        x1, y1, x2, y2, _, d = row
        class_name = class_list[d]

        if 'raider' in class_name:
            raider_bboxes.append([x1, y1, x2, y2])

        elif 'nohelmet' in class_name:
            nohelmet_bboxes.append([x1, y1, x2, y2])

        elif 'helmet' in class_name:
            helmet_bboxes.append([x1, y1, x2, y2])

    bbox1_idx = tracker1.update(raider_bboxes)
    bbox2_idx = tracker2.update(nohelmet_bboxes)
    bbox3_idx = tracker3.update(helmet_bboxes)

    for bbox1 in bbox1_idx:
        x, y, w, h, id1 = bbox1
        cym = (y + h) // 2
        if cy1 - offset < cym < cy1 + offset:
            draw_rectangle_and_text(frame, bbox1, 'raider', (255, 255, 255))
            if id1 not in counter_raider:
                counter_raider.append(id1)

    for bbox2 in bbox2_idx:
        x, y, w, h, id2 = bbox2
        cym = (y + h) // 2
        if cy1 - offset < cym < cy1 + offset:
            draw_rectangle_and_text(frame, bbox2, 'nohelmet', (255, 255, 255))
            if id2 not in counter_nohelmet:
                counter_nohelmet.append(id2)

    for bbox3 in bbox3_idx:
        x, y, w, h, id3 = bbox3
        cym = (y + h) // 2
        if cy1 - offset < cym < cy1 + offset:
            draw_rectangle_and_text(frame, bbox3, 'helmet', (255, 255, 255))
            if id3 not in counter_helmet:
                counter_helmet.append(id3)

    return frame, counter_raider, counter_nohelmet, counter_helmet

counter_raider = []
counter_nohelmet = []
counter_helmet = []

current_raider_count = 0
current_nohelmet_count = 0
current_helmet_count = 0

def generate_frame():
    global counter_raider, counter_nohelmet, counter_helmet, current_raider_count, current_nohelmet_count, current_helmet_count
    path = 'videoempat.mp4'
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    model = YOLO('model/Model2.pt')

    my_file = open("class/coco.txt", "r")
    class_list = my_file.read().split("\n")

    tracker1 = Tracker()
    tracker2 = Tracker()
    tracker3 = Tracker()

    offset = 6
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame, counter_raider, counter_nohelmet, counter_helmet = objectcount(
            frame, model, class_list, tracker1, tracker2, tracker3, counter_raider, counter_nohelmet, counter_helmet
        )

        # Draw the counting line
        cy1 = 424
        cv2.line(frame, (2, cy1), (1300, cy1), (0, 230, 255), 10)

        current_raider_count = len(counter_raider)
        current_nohelmet_count = len(counter_nohelmet)
        current_helmet_count = len(counter_helmet)

        # ...
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route("/", methods=['GET', 'POST'])
def webcam():
    global current_raider_count, current_nohelmet_count, current_helmet_count
    return render_template('ui.html', raider_counter=current_raider_count, nohelmet_counter=current_nohelmet_count, helmet_counter=current_helmet_count)

@app.route('/get_counts')
def get_count():
    global current_raider_count, current_nohelmet_count, current_helmet_count
    return {
        'raider_count': current_raider_count,
        'nohelmet_count': current_nohelmet_count,
        'helmet_count': current_helmet_count,
    }


@app.route('/webapp')
def webapp():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
