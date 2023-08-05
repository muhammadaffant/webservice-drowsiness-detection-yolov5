import torch
import numpy as np
import cv2
import vlc
import random
from PIL import Image
from flask_restx import Resource, Api, reqparse
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import jwt
import os
from flask import Flask, render_template, make_response, Response, jsonify, session
from flask_socketio import SocketIO
from flask_mail import Mail, Message
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import functools
import base64
import operator

# time
# import threading
import time

from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
socketio = SocketIO(app)
api = Api(app)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:@127.0.0.1:3306/webservice"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'WhatEverYouWant'
app.config['MAIL_SERVER'] = 'smtp.gmail.com' 
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
mail = Mail(app)
db = SQLAlchemy(app) 


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    waktu = db.Column(db.DateTime, default=datetime.now)
    label = db.Column(db.String(10))

    def __repr__(self):
        return f"<Detection(id={self.id}, waktu={self.waktu}, label={self.label})>"


class Users(db.Model):
    id          = db.Column(db.Integer(), primary_key=True, nullable=False)
    username   = db.Column(db.String(64), nullable=False)
    email       = db.Column(db.String(32), unique=True, nullable=False)
    password    = db.Column(db.String(256), nullable=False)
    verified = db.Column(db.Boolean(1),nullable=False)
    created_at  = db.Column(db.Date)
    updated_at   = db.Column(db.Date)

# Registrasi
regParser = reqparse.RequestParser()
regParser.add_argument('username', type=str, help='Username', location='json', required=True)
regParser.add_argument('email', type=str, help='Email', location='json', required=True)
regParser.add_argument('password', type=str, help='Password', location='json', required=True)
regParser.add_argument('re_password', type=str, help='Retype Password', location='json', required=True)


@api.route('/signup')
class Regis(Resource):
    @api.expect(regParser)
    def post(self):

        args = regParser.parse_args()
        username = args['username']
        email = args['email']
        password = args['password']
        rePassword = args['re_password']
        verified = False

        if password != rePassword:
            return {
                'messege': 'Password tidak cocok'
            }, 400

        user = db.session.execute(
            db.select(Users).filter_by(email=email)).first()
        if user:
            return "Email sudah terpakai silahkan coba lagi menggunakan email lain"
        # END: Check email existance.

        # BEGIN: Insert new user.
        user = Users()  # Instantiate User object.
        user.username = username
        user.email = email
        user.password = generate_password_hash(password)
        user.verified = verified
        db.session.add(user)
        msg = Message(subject='Verification OTP',sender=os.environ.get("MAIL_USERNAME"),recipients=[user.email])
        token =  random.randrange(10000,99999)
        session['email'] = user.email
        session['token'] = str(token)
        msg.html=render_template(
        'verify_email.html', token=token)
        mail.send(msg)
        db.session.commit()
        # END: Insert new user.
        return {'messege': 'Registrasi Berhasil, Cek email anda untuk verifikasi!'}, 200


# Verifikasi
otpparser = reqparse.RequestParser()
otpparser.add_argument('otp', type=str, help='otp', location='json', required=True)


@api.route('/verifikasi')
class Verifikasi(Resource):
    @api.expect(otpparser)
    def post(self):
        args = otpparser.parse_args()
        otp = args['otp']
        if 'token' in session:
            sesion = session['token']
            if otp == sesion:
                email = session['email']
                user = Users.query.filter_by(email=email).first()
                user.verified = True
                db.session.commit()
                session.pop('token', None)
                return {'message': 'Email berhasil diverifikasi'}
            else:
                return {'message': 'Kode Otp tidak sesuai'}
        else:
            return {'message': 'Kode Otp tidak sesuai'}


# login
logParser = reqparse.RequestParser()
logParser.add_argument('email', type=str, help='Email',
                       location='json', required=True)
logParser.add_argument('password', type=str,
                       help='Password', location='json', required=True)

SECRET_KEY = "WhatEverYouWant"
ISSUER = "myFlaskWebservice"
AUDIENCE_MOBILE = "myMobileApp"


@api.route('/signin')
class LogIn(Resource):
    @api.expect(logParser)
    def post(self):
        # BEGIN: Get request parameters.
        args = logParser.parse_args()
        email = args['email']
        password = args['password']
        # END: Get request parameters.

        if not email or not password:
            return {
                'message': 'Silakan isi email dan kata sandi Anda'
            }, 400

        # BEGIN: Check email existance.
        user = db.session.execute(
            db.select(Users).filter_by(email=email)).first()

        if not user:
            return {
                'message': 'Email atau kata sandi salah'
            }, 400
        else:
            user = user[0]  # Unpack the array.
        # END: Check email existance.

        # BEGIN: Check password hash.
        if check_password_hash(user.password, password):
            payload = {
                'user_id': user.id,
                'user_email': user.email,
                'aud': AUDIENCE_MOBILE,  # AUDIENCE_WEB
                'iss': ISSUER,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=4)
            }
            tup = email,":",password
            #toString
            data = functools.reduce(operator.add, tup)
            byte_msg = data.encode('ascii')
            base64_val = base64.b64encode(byte_msg)
            code = base64_val.decode('ascii')
            token = jwt.encode(payload, SECRET_KEY)
            msg_title = "KODE APLIKASI"
            msg = Message(msg_title, sender='affanuppan@gmail.com',
                          recipients=[email])
            msg.body = token
            mail.send(msg)
            return {
                'token': token,
                'code': code
            }, 200
        else:
            return {
                'message': 'Email atau password salah'
            }, 400

        # END: Check password hash.


def decodetoken(jwtToken):
    decode_result = jwt.decode(
        jwtToken,
        SECRET_KEY,
        audience=[AUDIENCE_MOBILE],
        issuer=ISSUER,
        algorithms=['HS256'],
        options={"require": ["aud", "iss", "iat", "exp"]}
    )
    return decode_result

authParser = reqparse.RequestParser()
authParser.add_argument('Authorization', type=str, help='Authorization', location='headers', required=True)


@api.route('/user')
class DetailUser(Resource):
    @api.expect(authParser)
    def get(self):
        args = authParser.parse_args()
        bearerAuth = args['Authorization']
        try:
            jwtToken = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = db.session.execute(
                db.select(Users).filter_by(email=token['user_email'])).first()
            user = user[0]

              # Mengubah nilai verified pengguna menjadi True
            user.verified = True
            db.session.commit()
            
            data = {
                'username': user.username,
                'email': user.email
            }
        except:
            return {
                'message': 'Token Tidak valid, Silahkan Login Terlebih Dahulu'
            }, 401

        return data, 200


# editpassword
editPasswordParser = reqparse.RequestParser()
editPasswordParser.add_argument(
    'current_password', type=str, help='current_password', location='json', required=True)
editPasswordParser.add_argument(
    'new_password', type=str, help='new_password', location='json', required=True)


@api.route('/editpassword')
class EditPassword(Resource):
    @api.expect(authParser, editPasswordParser)
    def put(self):
        args = editPasswordParser.parse_args()
        argss = authParser.parse_args()
        bearerAuth = argss['Authorization']
        cu_password = args['current_password']
        newpassword = args['new_password']
        try:
            jwtToken = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = Users.query.filter_by(id=token.get('user_id')).first()
            if check_password_hash(user.password, cu_password):
                user.password = generate_password_hash(newpassword)
                db.session.commit()
            else:
                return {'message': 'Password Lama Salah'}, 400
        except:
            return {
                'message': 'Token Tidak valid, Silahkan Login Terlebih Dahulu'
            }, 401
        return {'message': 'Password Berhasil Diubah'}, 200


# Edit Users
editParser = reqparse.RequestParser()
editParser.add_argument('username', type=str,
                        help='Username', location='json', required=True)
editParser.add_argument('Authorization', type=str,
                        help='Authorization', location='headers', required=True)


@api.route('/edituser')
class EditUser(Resource):
    @api.expect(editParser)
    def put(self):
        args = editParser.parse_args()
        bearerAuth = args['Authorization']
        username = args['username']
        datenow = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        try:
            jwtToken = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = Users.query.filter_by(email=token.get('user_email')).first()
            user.username = username
            user.updatedAt = datenow
            db.session.commit()
        except:
            return {
                'message': 'Token Tidak valid, Silahkan Login Terlebih Dahulu'
            }, 401
        return {'message': 'Update User Suksess'}, 200


# LOAD MODEL
# FUNGSI MODEL

# @app.route('/home')
# def home():
#     return render_template('index.html')

@app.route('/detection')
def detection_er():
    return render_template('detection.html', counter=counter)


# @app.route('/load_model', methods=['POST'])
# def load_model():
#     global model
#     model_path = 'best.pt'  # Ganti dengan jalur yang sesuai ke file model Anda
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
#     return "Model loaded"

@app.route('/load_model', methods=['POST'])
def load_model():
    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model yolo/runs/train/exp20/weights/best.pt', force_reload=True)
    return "Model loaded"

counter = 0
model = None
cap = None

# def reset_counter():
#     global counter
#     counter = 0


def detect():
    global counter
    global model
    global cap
    start_time = None

    if model is None:
        load_model()

    if cap is None:
        # camera = 'http://192.168.43.76/video'
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        img = np.squeeze(results.render())

        if len(results.xywh[0]) > 0:
            dconf = results.xywh[0][0][4]
            dclass = results.xywh[0][0][5]

            #jika pada class 1 (terlelap) dengan tingkat kepercayaan 0.8 lalu akan memutar file audio
            if dconf.item() > 0.8 and dclass.item() == 1.0:
                filechoice = random.choice([1, 2, 3])
                p = vlc.MediaPlayer(f"file:///{filechoice}.wav")
                p.play()
                counter += 1
                start_time = time.time()  # Catat waktu deteksi pertama

                # jika angka counter mencapai 30 maka akan menyimpan ke database
                if counter == 10:
                    with app.app_context():
                        # Simpan ke database
                        detection = Detection(label="terlelap")
                        db.session.add(detection)
                        db.session.commit()
                        
                    # jika counter sudah mencapai 30 maka fungsi ini untuk mereset kembali ke 0
                    # Reset counter
                    counter = 0
            elif dclass.item() == 0.0:
                if start_time is not None and time.time() - start_time >= 20:
                    with app.app_context():
                        # Simpan ke database
                        detection = Detection(label="terjaga")
                        db.session.add(detection)
                        db.session.commit()
                    counter = 0
                else:
                    counter = 0

            # jika terdeteksi pada label class 0 (terjaga) maka angka counter otomatis menjadi 0 
            elif dclass.item() == 0.0:
                counter = 0

        imgarr = Image.fromarray(img)
        
        # Mengirim hasil deteksi sebagai respons JSON
        # yield jsonify({'frame': jpeg.tobytes(), 'counter': counter})

        # Convert image to JPEG format
        frame = cv2.cvtColor(np.array(imgarr), cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed')
# def video_feed():
#     return Response(detect(), mimetype='text/event-stream')

# mengambil jumlah counter
@app.route('/get_counter')
def get_counter():
    global counter
    return str(counter)

# melihat data yang tersimpan di database jika terdeteksi terlelap
@app.route('/data', methods=['GET'])
def get_detections():
    detections = Detection.query.all()
    return render_template('data.html', detections=detections)

if __name__ == '__main__':
    app.run( host='192.168.43.76', port=5000, debug=True)