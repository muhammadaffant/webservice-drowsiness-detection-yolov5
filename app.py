import torch
import numpy as np
import cv2
import vlc
import random
from PIL import Image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

counter = 0
model = None
cap = None

def reset_counter():
    global counter
    counter = 0
    socketio.emit('counter_update', counter)  # Emit counter value to connected clients

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection_er():
    return render_template('detection.html', counter=counter)

def load_model():
    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp20/weights/best.pt', force_reload=True)

def detect():
    global counter
    global model
    global cap

    if model is None:
        load_model()

    if cap is None:
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        img = np.squeeze(results.render())

        if len(results.xywh[0]) > 0:
            dconf = results.xywh[0][0][4]
            dclass = results.xywh[0][0][5]

            if dconf.item() > 0.8 and dclass.item() == 1.0:
                filechoice = random.choice([1, 2, 3])
                p = vlc.MediaPlayer(f"file:///{filechoice}.wav")
                p.play()
                counter += 1
                socketio.emit('counter_update', counter)  # Emit counter value to connected clients

            if dclass.item() == 0.0:
                counter = 0
                socketio.emit('counter_update', counter)  # Emit counter value to connected clients

        imgarr = Image.fromarray(img)

        # Convert image to JPEG format
        frame = cv2.cvtColor(np.array(imgarr), cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@socketio.on('connect')
def handle_connect():
    socketio.emit('counter_update', counter)  # Emit initial counter value to connected clients

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_counter')
def reset_counter_route():
    reset_counter()
    return "Counter reset"

if __name__ == '__main__':
    socketio.run(app, host='192.168.43.56', port=5000, debug=True)
