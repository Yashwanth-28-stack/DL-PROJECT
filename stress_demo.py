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

                if confidence >= MIN_CONFIDENCE:
                    votes[emotion] += 1
                    confidences.setdefault(emotion, []).append(confidence)
                    label = f"{emotion}: {confidence:.0%}"
                else:
                    label = "Uncertain"

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 220, 0), 2
                )
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 0),
                    2
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