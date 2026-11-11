import os
import time
import smtplib
from collections import Counter
from email.message import EmailMessage

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "emotion_vgg.keras"
SECONDS = 15
MIN_CONFIDENCE = 0.40

CLASSES = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

LEVELS = {
    "happy": "Low",
    "surprise": "Low",
    "neutral": "Moderate",
    "disgust": "Moderate",
    "sad": "High",
    "angry": "High",
    "fear": "High",
}

# OpenCV uses BGR colors.
EMOTION_COLORS = {
    "angry": (70, 70, 240),       # red
    "disgust": (190, 80, 190),    # purple
    "fear": (0, 145, 255),        # orange
    "happy": (0, 220, 255),       # yellow
    "neutral": (220, 180, 100),   # light blue
    "sad": (50, 60, 255),         # red
    "surprise": (100, 220, 100),  # green
}
UNCERTAIN_COLOR = (150, 160, 170)


def send_email_alert(emotion, confidence):
    message = EmailMessage()
    message["Subject"] = "Facial expression demo alert"
    message["From"] = os.environ["GMAIL_ADDRESS"]
    message["To"] = os.environ["ALERT_TO"]
    message.set_content(
        f"The webcam demo assigned a High level based on its most frequent "
        f"expression prediction: {emotion} "
        f"(emotion prediction confidence: {confidence:.0%}).\n\n"
        "This is an experimental project rule, not a validated stress assessment."
    )

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.starttls()
        server.login(
            os.environ["GMAIL_ADDRESS"],
            os.environ["GMAIL_APP_PASSWORD"]
        )
        server.send_message(message)


def draw_confidence_panel(frame, label, confidence, color):
    height, width = frame.shape[:2]
    panel_x, panel_y = 15, 55
    panel_width = min(350, width - 30)
    panel_height = 112

    if panel_width < 180 or height < panel_y + panel_height:
        return frame

    # Translucent dark panel.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (18, 26, 38),
        -1
    )
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    cv2.putText(
        frame, "LIVE EMOTION", (panel_x + 15, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 205, 220), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, label, (panel_x + 15, panel_y + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA
    )

    confidence_text = f"{confidence:.0%}"
    text_size = cv2.getTextSize(
        confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2
    )[0]
    cv2.putText(
        frame, confidence_text,
        (panel_x + panel_width - 15 - text_size[0], panel_y + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 248, 252), 2, cv2.LINE_AA
    )

    bar_x = panel_x + 15
    bar_y = panel_y + 78
    bar_width = panel_width - 30
    bar_height = 18
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
        (65, 76, 91), -1
    )

    fill_width = int(bar_width * min(max(confidence, 0.0), 1.0))
    if fill_width > 0:
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
            color, -1
        )

    return frame


def main():
    for variable in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "ALERT_TO"):
        if not os.environ.get(variable):
            print(f"Set the {variable} environment variable first.")
            return

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    cascade_path = os.path.join(
        os.path.dirname(cv2.__file__),
        "data",
        "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    camera = cv2.VideoCapture(0)

    if detector.empty() or not camera.isOpened():
        print("Could not start face detection or webcam.")
        camera.release()
        return

    votes = Counter()
    confidences = {}
    start = time.time()

    print(f"Webcam session: {SECONDS} seconds. Press Q to cancel.")

    try:
        while time.time() - start < SECONDS:
            ok, frame = camera.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            live_label = "No face detected"
            live_confidence = 0.0
            live_color = UNCERTAIN_COLOR

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                face = face.astype("float32")[None, ..., None] / 255.0

                probabilities = model.predict(face, verbose=0)[0]
                index = int(np.argmax(probabilities))
                emotion = CLASSES[index]
                confidence = float(probabilities[index])
                emotion_color = EMOTION_COLORS[emotion]

                if confidence >= MIN_CONFIDENCE:
                    votes[emotion] += 1
                    confidences.setdefault(emotion, []).append(confidence)
                    live_label = emotion.title()
                    live_color = emotion_color
                    face_label = f"{emotion}: {confidence:.0%}"
                else:
                    live_label = f"Uncertain ({emotion})"
                    live_color = UNCERTAIN_COLOR
                    face_label = f"Uncertain: {confidence:.0%}"

                live_confidence = confidence

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), emotion_color, 2
                )
                cv2.putText(
                    frame,
                    face_label,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    emotion_color,
                    2,
                    cv2.LINE_AA
                )

            frame = draw_confidence_panel(
                frame, live_label, live_confidence, live_color
            )

            remaining = max(0, SECONDS - int(time.time() - start))
            cv2.putText(
                frame,
                f"{remaining}s | Q to quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 220),
                2
            )
            cv2.imshow("Emotion demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Cancelled; no email sent.")
                return

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not votes:
        print("No confident predictions collected; no email sent.")
        return

    emotion, count = votes.most_common(1)[0]
    level = LEVELS[emotion]
    confidence = sum(confidences[emotion]) / len(confidences[emotion])

    print(
        f"Emotion: {emotion} | Assigned demo level: {level} "
        f"| Emotion confidence: {confidence:.0%}"
    )
    print(f"Frames voting for this emotion: {count}")

    if level == "High":
        try:
            send_email_alert(emotion, confidence)
            print("High-level email sent.")
        except Exception as error:
            print("Email failed:", error)
    else:
        print("No email sent; assigned demo level was not High.")

    print("The assigned level is a project rule, not a validated stress assessment.")


if __name__ == "__main__":
    main()
