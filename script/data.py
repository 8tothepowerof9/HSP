import cv2
import mediapipe as mp
import pandas as pd
import os
import time
from utils import read_command

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

dataset_path = "data"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)


def capture_data(letter, num_samples=200, delay=0.05):
    cap = cv2.VideoCapture(0)
    data = []
    collecting = False
    paused = False
    last_capture_time = time.time()
    next_pause_threshold = int(0.5 * num_samples)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                now = time.time()
                if collecting and not paused and (now - last_capture_time) >= delay:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    data.append(landmarks)
                    last_capture_time = now

                    if len(data) >= next_pause_threshold and len(data) < num_samples:
                        paused = True
                        next_pause_threshold += int(0.5 * num_samples)
                        print("Auto-paused at", len(data), "samples")

        status_text = "Status: "
        if not collecting:
            status_text += "Waiting to start"
        elif paused:
            status_text += "Paused"
        else:
            status_text += f"Collecting ({len(data)}/{num_samples})"

        cv2.rectangle(frame, (20, 20), (400, 60), (0, 255, 0), -1)
        cv2.putText(
            frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        cv2.imshow("ASL Data Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            collecting = True
            paused = False
            last_capture_time = time.time()
            print(f"Collecting data for letter '{letter}'...")
        elif key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord("q"):
            break

        if collecting and not paused and len(data) >= num_samples:
            print(f"Collected {num_samples} samples for '{letter}'")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(data) == 0:
        print("No data collected!")
    else:
        df = pd.DataFrame(data)
        csv_path = f"{dataset_path}/{letter}.csv"
        df.to_csv(csv_path, index=False, header=False)
        print(f"Saved {num_samples} samples for letter '{letter}' at {csv_path}")


if __name__ == "__main__":
    args = read_command()
    capture_data(args.character, args.num_samples)
