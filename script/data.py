import cv2
import mediapipe as mp
import pandas as pd
import os
import time
from .utils import read_command

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Create a folder to store data
dataset_path = "data"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)


def capture_data(letter, delay=0.1, num_samples=200):
    """Captures ASL hand landmarks and displays them on screen."""
    cap = cv2.VideoCapture(0)  # Use camera index 0
    data = []  # Store landmark data
    collecting = False  # Wait until button is pressed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the screen
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if collecting:
                    # Extract and save landmark coordinates
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    data.append(landmarks)
                    time.sleep(delay)

        # Draw a start button on the screen
        button_text = "Press 'S' to Start, 'Q' to Quit"
        cv2.rectangle(frame, (20, 20), (300, 60), (0, 255, 0), -1)  # Green button
        cv2.putText(
            frame, button_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        cv2.imshow("ASL Data Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 'S' to start capturing
            collecting = True
            print(f"Collecting data for letter '{letter}'...")

        if key == ord("q"):  # Press 'Q' to quit
            break

        # Stop collecting after num_samples
        if collecting and len(data) >= num_samples:
            print(f"Collected {num_samples} samples for '{letter}'")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    csv_path = f"{dataset_path}/{letter}.csv"
    df.to_csv(csv_path, index=False, header=False)
    print(f"Saved {num_samples} samples for letter '{letter}' at {csv_path}")


if __name__ == "__main__":
    args = read_command()
    capture_data(args.character, args.delay, args.num_samples)
