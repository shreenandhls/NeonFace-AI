
import cv2
import time
import collections
import numpy as np

from flask import Flask, render_template, request, Response, jsonify
from age_detection import detect_and_predict, draw_cool_ui

app = Flask(__name__)

current_stats = {
    "total_faces": 0,
    "fps": 0,
    "logs": []
}

age_history = collections.deque(maxlen=12)
gender_history = collections.deque(maxlen=12)

last_frame_time = time.time()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global last_frame_time

    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]
    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "Empty frame"}), 400

    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Resize frame for processing
    frame = cv2.resize(frame, (640, 480))

    face_predictions = detect_and_predict(frame)

    current_logs = []

    for pred in face_predictions or []:
        x1, y1, x2, y2 = pred["bbox"]

        age = pred["age"]
        gender = pred["gender"]

        age_conf = float(pred["age_conf"])
        gender_conf = float(pred["gender_conf"])

        age_history.append(age)
        gender_history.append(gender)

        if len(age_history) >= 4:
            age_result = max(set(age_history), key=age_history.count)
            gender_result = max(set(gender_history), key=gender_history.count)
        else:
            age_result = age
            gender_result = gender

        label = f"{gender_result}, {age_result}"

        box_color = (
            (200, 100, 255)
            if gender_result == "Female"
            else (0, 255, 255)
        )

        draw_cool_ui(
            frame,
            x1, y1, x2, y2,
            label,
            box_color
        )

        current_logs.append({
            "gender": gender_result,
            "age": age_result,
            "age_conf": age_conf,
            "gender_conf": gender_conf
        })

    # Calculate processing FPS
    current_time = time.time()
    elapsed = current_time - last_frame_time

    fps = 1 / elapsed if elapsed > 0 else 0
    last_frame_time = current_time

    current_stats["total_faces"] = len(face_predictions or [])
    current_stats["fps"] = round(fps, 1)
    current_stats["logs"] = current_logs

    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        return jsonify({"error": "Frame encoding failed"}), 500

    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg"
    )


@app.route("/stats")
def stats():
    return jsonify(current_stats)


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    age_history.clear()
    gender_history.clear()

    current_stats["total_faces"] = 0
    current_stats["fps"] = 0
    current_stats["logs"] = []

    return jsonify({"status": "stopped"})


@app.route("/start_camera", methods=["POST"])
def start_camera():
    return jsonify({"status": "started"})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
