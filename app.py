from flask import Flask, render_template, request
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

app = Flask(__name__)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

VIDEO_UPLOAD_FOLDER = "static/videos"
FRAME_FOLDER = "static/frames"

app.config["VIDEO_UPLOAD_FOLDER"] = VIDEO_UPLOAD_FOLDER
app.config["FRAME_FOLDER"] = FRAME_FOLDER

os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

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


def get_behavior_timeline():
    log_path = "logs/behavior_log.csv"

    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        logs = list(reader)

    return logs[-10:]


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

def get_animal_behavior_summary():
    log_path = "logs/behavior_log.csv"

    animal_summary = {}

    if not os.path.exists(log_path):
        return animal_summary

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            animal_id = row["animal_id"]
            behavior = row["behavior"]

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


def create_observation_prompt(summary, abnormal_message, animal_behavior_summary):
    summary_text = create_summary_text(summary)

    prompt = f"""
あなたは動物園で飼育されているカピバラの
観察レポートを作成するAIアシスタントです。

対象動物：
カピバラ

このレポートは飼育員や研究担当者が
日々の行動傾向を把握するために利用します。

以下の行動ログ集計と分析結果を参考に、
自然な日本語で観察レポートを作成してください。

条件：

・観察結果を要約する
・行動割合から考えられる傾向を説明する
・異常の可能性がある場合は断定しない
・必要に応じて継続観察を提案する
・読みやすい文章で作成する

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
        response = client.responses.create(
            model="gpt-5",
            input=prompt
        )

        return response.output_text

    except Exception as e:
        return f"AIレポート生成エラー: {str(e)}"


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


def predict_behavior(frame_path):
    img = Image.open(frame_path).convert("RGB")
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


@app.route("/")
def index():
    recent_logs = get_recent_logs()
    behavior_timeline = get_behavior_timeline()
    behavior_summary = get_behavior_summary()
    animal_behavior_summary = get_animal_behavior_summary()
    abnormal_message = analyze_abnormal_behavior(behavior_summary)

    observation_prompt = create_observation_prompt(
        behavior_summary,
        abnormal_message,
        animal_behavior_summary
    )

    return render_template(
        "index.html",
        recent_logs=recent_logs,
        behavior_timeline=behavior_timeline,
        behavior_summary=behavior_summary,
        animal_behavior_summary=animal_behavior_summary,
        abnormal_message=abnormal_message,
        observation_prompt=observation_prompt
    )


@app.route("/clear-logs", methods=["POST"])
def clear_logs():
    log_path = "logs/behavior_log.csv"

    if os.path.exists(log_path):
        os.remove(log_path)

    for filename in os.listdir(app.config["VIDEO_UPLOAD_FOLDER"]):
        file_path = os.path.join(
            app.config["VIDEO_UPLOAD_FOLDER"],
            filename
        )

        if os.path.isfile(file_path):
            os.remove(file_path)

    for filename in os.listdir(app.config["FRAME_FOLDER"]):
        file_path = os.path.join(
            app.config["FRAME_FOLDER"],
            filename
        )

        if os.path.isfile(file_path):
            os.remove(file_path)

    return render_template(
        "index.html",
        message="現在の行動ログを削除しました。",
        recent_logs=[],
        behavior_timeline=[],
        behavior_summary={
            "睡眠": 0,
            "食事": 0,
            "陸上活動": 0,
            "水中活動": 0
        },
        abnormal_message="十分な行動ログがありません。"
    )


@app.route("/generate-report", methods=["POST"])
def generate_report():
    recent_logs = get_recent_logs()
    behavior_timeline = get_behavior_timeline()
    behavior_summary = get_behavior_summary()
    animal_behavior_summary = get_animal_behavior_summary()
    abnormal_message = analyze_abnormal_behavior(behavior_summary)

    observation_prompt = create_observation_prompt(
        behavior_summary,
        abnormal_message,
        animal_behavior_summary
    )

    if sum(behavior_summary.values()) > 0:
        ai_report = generate_ai_report(
            observation_prompt
        )
    else:
        ai_report = None

    return render_template(
        "index.html",
        recent_logs=recent_logs,
        behavior_timeline=behavior_timeline,
        behavior_summary=behavior_summary,
        abnormal_message=abnormal_message,
        observation_prompt=observation_prompt,
        ai_report=ai_report,
        animal_behavior_summary=animal_behavior_summary
    )


@app.route("/session", methods=["POST"])
def create_session():
    animal_id = request.form["animal_id"]

    if animal_id.strip() == "":
        return render_template(
            "index.html",
            message="個体IDを入力してください。",
            recent_logs=get_recent_logs(),
            behavior_timeline=get_behavior_timeline(),
            behavior_summary=get_behavior_summary(),
            abnormal_message=analyze_abnormal_behavior(
                get_behavior_summary()
            )
        )

    start_time = request.form["start_time"]
    frame_interval = request.form["frame_interval"]

    if not frame_interval.isdigit() or int(frame_interval) <= 0:
        return render_template(
            "index.html",
            message="フレーム抽出間隔は1秒以上の数値を入力してください。",
            recent_logs=get_recent_logs(),
            behavior_timeline=get_behavior_timeline(),
            behavior_summary=get_behavior_summary(),
            animal_behavior_summary = get_animal_behavior_summary(),
            abnormal_message=analyze_abnormal_behavior(get_behavior_summary())
        )


    video = request.files.get("video")

    if video is None or video.filename == "":
        return render_template(
            "index.html",
            message="動画ファイルを選択してください。",
            recent_logs=get_recent_logs(),
            behavior_timeline=get_behavior_timeline(),
            behavior_summary=get_behavior_summary(),
            animal_behavior_summary=get_animal_behavior_summary(),
            abnormal_message=analyze_abnormal_behavior(get_behavior_summary())
        )

    video_path = os.path.join(
        app.config["VIDEO_UPLOAD_FOLDER"],
        video.filename
    )

    video.save(video_path)

    saved_frames = extract_frames_from_video(
        video_path,
        frame_interval
    )

    log_path = "logs/behavior_log.csv"
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "datetime",
                "animal_id",
                "start_time",
                "frame_number",
                "behavior",
                "confidence"
            ])

        for frame_info in saved_frames:
            frame_path = frame_info["path"]
            frame_number = frame_info["frame_number"]
            fps = frame_info["fps"]

            label, confidence = predict_behavior(frame_path)

            real_time = calculate_real_time(
                start_time,
                frame_number,
                fps
            )

            frame_filename = os.path.basename(frame_path)

            writer.writerow([
                real_time,
                animal_id,
                start_time,
                frame_filename,
                label,
                round(float(confidence) * 100, 2)
            ])

    recent_logs = get_recent_logs()
    behavior_timeline = get_behavior_timeline()
    behavior_summary = get_behavior_summary()
    animal_behavior_summary = get_animal_behavior_summary()
    abnormal_message = analyze_abnormal_behavior(behavior_summary)

    observation_prompt = create_observation_prompt(
        behavior_summary,
        abnormal_message,
        animal_behavior_summary
    )

    return render_template(
        "index.html",
        message=f"観察セッションを作成しました。抽出フレーム数：{len(saved_frames)}枚",
        animal_id=animal_id,
        start_time=start_time,
        frame_interval=frame_interval,
        video_filename=video.filename,
        frame_count=len(saved_frames),
        recent_logs=recent_logs,
        behavior_timeline=behavior_timeline,
        behavior_summary=behavior_summary,
        animal_behavior_summary=animal_behavior_summary,
        abnormal_message=abnormal_message,
        observation_prompt=observation_prompt
    )


if __name__ == "__main__":
    app.run(debug=True)