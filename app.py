from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import csv
import cv2
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import sys
import uuid

app = Flask(__name__)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

IMAGE_UPLOAD_FOLDER = "static/uploads"
VIDEO_UPLOAD_FOLDER = "static/videos"
FRAME_FOLDER = "static/frames"
LOG_PATH = "logs/behavior_log.csv"

app.config["IMAGE_UPLOAD_FOLDER"] = IMAGE_UPLOAD_FOLDER
app.config["VIDEO_UPLOAD_FOLDER"] = VIDEO_UPLOAD_FOLDER
app.config["FRAME_FOLDER"] = FRAME_FOLDER

os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

model = load_model("model/keras_model.h5")

with open("model/labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]


CSV_COLUMNS = [
    "log_id",
    "session_id",
    "datetime",
    "animal_id",
    "image_name",
    "behavior",
    "confidence"
]


BEHAVIOR_LIST = [
    "睡眠",
    "食事",
    "陸上活動",
    "水中活動"
]


def read_behavior_logs():
    if not os.path.exists(LOG_PATH):
        return []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        logs = list(reader)

    return logs


def write_behavior_logs(logs):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        writer.writeheader()

        for log in logs:
            writer.writerow(log)


def get_available_animals():
    logs = read_behavior_logs()
    animal_ids = []

    for log in logs:
        animal_id = log["animal_id"]

        if animal_id not in animal_ids:
            animal_ids.append(animal_id)

    return animal_ids


def filter_logs_by_animal(logs, selected_animal_id):
    if selected_animal_id == "all":
        return logs

    return [
        log for log in logs
        if log["animal_id"] == selected_animal_id
    ]


def get_recent_logs():
    logs = read_behavior_logs()

    return logs[-8:]


def get_behavior_timeline(selected_animal_id="all"):
    logs = read_behavior_logs()
    filtered_logs = filter_logs_by_animal(logs, selected_animal_id)

    return filtered_logs[-12:]


def get_behavior_summary(selected_animal_id="all"):
    logs = read_behavior_logs()
    filtered_logs = filter_logs_by_animal(logs, selected_animal_id)

    summary = {
        "睡眠": 0,
        "食事": 0,
        "陸上活動": 0,
        "水中活動": 0
    }

    for log in filtered_logs:
        behavior = log["behavior"]

        if behavior in summary:
            summary[behavior] += 1

    return summary


def get_animal_behavior_summary():
    logs = read_behavior_logs()

    animal_summary = {}

    for log in logs:
        animal_id = log["animal_id"]
        behavior = log["behavior"]

        if animal_id not in animal_summary:
            animal_summary[animal_id] = {
                "睡眠": 0,
                "食事": 0,
                "陸上活動": 0,
                "水中活動": 0
            }

        if behavior in animal_summary[animal_id]:
            animal_summary[animal_id][behavior] += 1

    return animal_summary


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


def create_observation_prompt(
    summary,
    abnormal_message,
    animal_behavior_summary,
    selected_animal_id
):
    summary_text = create_summary_text(summary)

    prompt = f"""
あなたは動物園で飼育されているカピバラの
観察レポートを作成するAIアシスタントです。

対象動物：
カピバラ

対象個体ID：
{selected_animal_id}

このレポートは飼育員や研究担当者が
日々の行動傾向を把握するために利用します。

以下の行動ログ集計と分析結果を参考に、
自然な日本語で観察レポートを作成してください。

条件：

・選択された個体IDの行動傾向を中心に説明する
・観察結果を要約する
・行動割合から考えられる傾向を説明する
・異常の可能性がある場合は断定しない
・必要に応じて継続観察を提案する
・読みやすい文章で作成する
・専門的すぎる表現は避ける
・飼育員がすぐ確認できるように簡潔にまとめる

出力形式：

【観察概要】
全体の行動傾向を2〜3文で要約してください。

【行動傾向】
睡眠、食事、陸上活動、水中活動の割合から読み取れる傾向を説明してください。

【異常傾向の確認】
異常の可能性がある場合は断定せず、「可能性があります」「継続観察が必要です」のように表現してください。

【個体別コメント】
対象個体IDの特徴を簡潔に説明してください。

【今後の観察ポイント】
次回以降、飼育員が確認すべきポイントを箇条書きで3つ以内にまとめてください。

【行動ログ集計】
{summary_text}

【異常傾向分析】
{abnormal_message}

【個体別行動サマリー】
{animal_behavior_summary}
"""

    return prompt


def generate_ai_report(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは動物園や研究機関向けに、カピバラの行動ログをもとに観察レポートを作成するAIアシスタントです。異常の可能性については断定せず、継続観察を前提に簡潔な日本語で説明してください。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AIレポート生成エラー: {str(e)}"


def predict_behavior(image_path):
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

    return label, confidence


def extract_frames_from_video(video_path, frame_interval):
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    interval_frames = int(fps * int(frame_interval))

    if interval_frames <= 0:
        interval_frames = 1

    frame_number = 0
    saved_frames = []
    max_frames = 10

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_number % interval_frames == 0:
            frame_filename = f"frame_{frame_number}.jpg"

            frame_path = os.path.join(
                app.config["FRAME_FOLDER"],
                frame_filename
            )

            cv2.imwrite(frame_path, frame)

            saved_frames.append({
                "path": frame_path,
                "frame_number": frame_number,
                "fps": fps
            })

            if len(saved_frames) >= max_frames:
                break

        frame_number += 1

    video.release()

    return saved_frames


def calculate_real_time(start_time, frame_number, fps):
    start_datetime = datetime.fromisoformat(start_time)
    elapsed_seconds = frame_number / fps
    real_time = start_datetime + timedelta(seconds=elapsed_seconds)

    return real_time.strftime("%Y-%m-%d %H:%M:%S")


def create_dashboard_context(selected_animal_id="all"):
    available_animals = get_available_animals()
    recent_logs = get_recent_logs()
    behavior_timeline = get_behavior_timeline(selected_animal_id)
    behavior_summary = get_behavior_summary(selected_animal_id)
    animal_behavior_summary = get_animal_behavior_summary()
    abnormal_message = analyze_abnormal_behavior(behavior_summary)

    observation_prompt = create_observation_prompt(
        behavior_summary,
        abnormal_message,
        animal_behavior_summary,
        selected_animal_id
    )

    return {
        "selected_animal_id": selected_animal_id,
        "available_animals": available_animals,
        "recent_logs": recent_logs,
        "behavior_timeline": behavior_timeline,
        "behavior_summary": behavior_summary,
        "animal_behavior_summary": animal_behavior_summary,
        "abnormal_message": abnormal_message,
        "observation_prompt": observation_prompt,
        "behavior_list": BEHAVIOR_LIST
    }


@app.route("/")
def index():
    selected_animal_id = request.args.get("animal_id", "all")
    message = request.args.get("message")

    context = create_dashboard_context(selected_animal_id)

    if message:
        context["message"] = message

    return render_template("index.html", **context)


@app.route("/image-session", methods=["POST"])
def create_image_session():
    animal_id = request.form["animal_id"]
    start_time = request.form["start_time"]
    interval_minutes = request.form["interval_minutes"]
    images = request.files.getlist("images")

    if animal_id.strip() == "":
        context = create_dashboard_context("all")
        context["message"] = "個体IDを入力してください。"
        return render_template("index.html", **context)

    if not interval_minutes.isdigit() or int(interval_minutes) < 0:
        context = create_dashboard_context("all")
        context["message"] = "画像間隔は0分以上の数値を入力してください。"
        return render_template("index.html", **context)

    if len(images) == 0 or images[0].filename == "":
        context = create_dashboard_context("all")
        context["message"] = "画像ファイルを選択してください。"
        return render_template("index.html", **context)

    start_datetime = datetime.fromisoformat(start_time)
    interval_minutes = int(interval_minutes)
    session_id = start_datetime.strftime("%Y%m%d%H%M%S")

    logs = read_behavior_logs()

    analyzed_count = 0
    latest_label = ""

    for index, image in enumerate(images):
        if image.filename == "":
            continue

        image_path = os.path.join(
            app.config["IMAGE_UPLOAD_FOLDER"],
            image.filename
        )

        image.save(image_path)

        label, confidence = predict_behavior(image_path)

        observation_datetime = start_datetime + timedelta(
            minutes=index * interval_minutes
        )

        log = {
            "log_id": str(uuid.uuid4())[:8],
            "session_id": session_id,
            "datetime": observation_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "animal_id": animal_id,
            "image_name": image.filename,
            "behavior": label,
            "confidence": round(float(confidence) * 100, 2)
        }

        logs.append(log)

        analyzed_count += 1
        latest_label = label

    write_behavior_logs(logs)

    message = f"{analyzed_count}枚の画像分析が完了しました。最新の判定結果：{latest_label}"

    return redirect(
        url_for(
            "index",
            animal_id=animal_id,
            message=message
        )
    )


@app.route("/update-behavior", methods=["POST"])
def update_behavior():
    log_id = request.form["log_id"]
    new_behavior = request.form["behavior"]
    selected_animal_id = request.form.get("selected_animal_id", "all")

    logs = read_behavior_logs()

    for log in logs:
        if log["log_id"] == log_id:
            log["behavior"] = new_behavior
            log["confidence"] = "手動修正"
            break

    write_behavior_logs(logs)

    return redirect(url_for("index", animal_id=selected_animal_id))


@app.route("/clear-logs", methods=["POST"])
def clear_logs():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    for folder in [
        app.config["IMAGE_UPLOAD_FOLDER"],
        app.config["VIDEO_UPLOAD_FOLDER"],
        app.config["FRAME_FOLDER"]
    ]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if os.path.isfile(file_path):
                os.remove(file_path)

    context = create_dashboard_context("all")
    context["message"] = "現在の行動ログを削除しました。"

    return render_template("index.html", **context)


@app.route("/generate-report", methods=["POST"])
def generate_report():
    selected_animal_id = request.form.get("selected_animal_id", "all")
    context = create_dashboard_context(selected_animal_id)

    if selected_animal_id == "all":
        context["ai_report"] = None
        context["message"] = "AI観察レポートを生成するには、個体IDを選択してください。"
    elif sum(context["behavior_summary"].values()) > 0:
        context["ai_report"] = generate_ai_report(
            context["observation_prompt"]
        )
    else:
        context["ai_report"] = None
        context["message"] = "選択された個体の行動ログがありません。"

    return render_template("index.html", **context)


@app.route("/session", methods=["POST"])
def create_session():
    context = create_dashboard_context("all")
    context["message"] = "デモ環境ではサーバー負荷を考慮し、画像分析機能を公開しています。"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)