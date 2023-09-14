from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        success, frames = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (a, b, y, z) in faces:

                cv2.rectangle(frames, (a, b), (a + y, b + z), (255, 0, 0), 2)
                roi = gray[b:b + z, a:a + y]
                eye = frames[b:b + z, a:a + y]
            eyes = eye_cascade.detectMultiScale(roi)
            for (ea, eb, ey, ez) in eyes:
                cv2.rectangle(eye, (ea, eb), (ea + ey, eb + ez), (0, 0, 255), 2)
            ret, buffer = cv2.imencode(".jpg", frames)
            frames = buffer.tobytes()
            yield b'--frames\r\nContent-type:/jpeg\r\n\r\n' + frames + b'\r\n'


@app.route("/")
def detect():
    return render_template("index555.html")


@app.route("/video")
def detect2():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frames')


app.run(debug=True)
