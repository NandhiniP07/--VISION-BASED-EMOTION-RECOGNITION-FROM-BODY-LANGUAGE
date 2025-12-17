import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# --- Load Model and Cascade ---
model = load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
EMOJI_DICT = {
    'angry': '😡', 'disgusted': '🤢', 'fearful': '😨',
    'happy': '😄', 'neutral': '😐', 'sad': '😢', 'surprised': '😲'
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

prev_time = 0
freeze = False
freeze_frame = None
freeze_start_time = 0

while True:
    if not freeze:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            face_pixels = roi_gray.astype('float32') / 255.0
            face_pixels = np.expand_dims(face_pixels, axis=0)
            face_pixels = np.expand_dims(face_pixels, axis=-1)

            prediction = model.predict(face_pixels, verbose=0)
            max_index = np.argmax(prediction[0])
            predicted_emotion = EMOTION_LABELS[max_index]
            emoji = EMOJI_DICT[predicted_emotion]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{predicted_emotion} {emoji}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    else:
        # Freeze: show the same frame
        frame = freeze_frame.copy()
        elapsed = time.time() - freeze_start_time
        cv2.putText(frame, f"Frozen for screenshot: {int(10-elapsed)}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        if elapsed >= 10:
            freeze = False

    cv2.imshow("Emotion Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not freeze:
        freeze = True
        freeze_frame = frame.copy()
        freeze_start_time = time.time()

cap.release()
cv2.destroyAllWindows()
