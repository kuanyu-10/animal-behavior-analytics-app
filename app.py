from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/keras_model.h5")

with open("model/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]


def get_recent_logs():
    log_path = "logs/behavior_log.csv"

    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        logs = list(reader)

    return logs[-5:]


def get_behavior_summary():
    log_path = "logs/behavior_log.csv"

    summary = {
        "睡眠": 0,
        "食事": 0,
        "陸上活動": 0,
        "水中活動": 0
    }

    if not os.path.exists(log_path):
        return summary

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            behavior = row["behavior"]

            if behavior in summary:
                summary[behavior] += 1

    return summary


def analyze_abnormal_behavior(summary):
    total = sum(summary.values())

    if total == 0:
        return "十分な行動ログがありません。"

    sleep_ratio = summary["睡眠"] / total

    if sleep_ratio >= 0.6:
        return "睡眠割合が高い可能性があります。継続的な観察を推奨します。"

    return "現在、明確な異常傾向は確認されていません。"


def create_summary_text(summary):

    total = sum(summary.values())

    if total == 0:
        return "行動ログがありません。"

    sleep_ratio = round(summary["睡眠"] / total * 100, 1)

    eating_ratio = round(summary["食事"] / total * 100, 1)

    land_ratio = round(summary["陸上活動"] / total * 100, 1)

    water_ratio = round(summary["水中活動"] / total * 100, 1)

    summary_text = f"""
総記録数：{total}

睡眠：{sleep_ratio}%

食事：{eating_ratio}%

陸上活動：{land_ratio}%

水中活動：{water_ratio}%
"""

    return summary_text


@app.route("/")
def index():
    recent_logs = get_recent_logs()
    behavior_summary = get_behavior_summary()
    abnormal_message = analyze_abnormal_behavior(behavior_summary)

    return render_template(
        "index.html",
        recent_logs=recent_logs,
        behavior_summary=behavior_summary,
        abnormal_message=abnormal_message
    )


@app.route("/upload", methods=["POST"])
def upload_image():
    image = request.files["image"]

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(image_path)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    english_label = labels[index].split(" ", 1)[1]

    label_map = {
        "sleeping": "睡眠",
        "eating": "食事",
        "land_activity": "陸上活動",
        "water_activity": "水中活動"
    }

    label = label_map[english_label]

    log_path = "logs/behavior_log.csv"

    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["datetime", "behavior", "confidence"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(float(confidence) * 100, 2)
        ])

    recent_logs = get_recent_logs()
    behavior_summary = get_behavior_summary()
    abnormal_message = analyze_abnormal_behavior(behavior_summary)

    image_url = url_for(
        "static",
        filename=f"uploads/{image.filename}"
    )

    return render_template(
        "index.html",
        message="画像をアップロードしました。",
        label=label,
        confidence=confidence,
        recent_logs=recent_logs,
        behavior_summary=behavior_summary,
        abnormal_message=abnormal_message,
        image_url=image_url
    )


if __name__ == "__main__":
    app.run(debug=True)