import cv2
import mediapipe as mp
import numpy as np
import time
import os
mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils
DATABASE_FILE = "sign_database.npz"
sign_database = {}
def load_database():
    global sign_database
    if os.path.exists(DATABASE_FILE):
        data = np.load(DATABASE_FILE, allow_pickle=True)
        sign_database = data["signs"].item()
        print(f"[INFO] Loaded {len(sign_database)} signs from database.")
    else:
        print("[INFO] No existing database found. Starting fresh.")

def save_database():
    np.savez(DATABASE_FILE, signs=sign_database)
    print("[INFO] Database saved successfully.")
def normalize_landmarks(landmarks):
    """Make landmarks invariant to position and scale."""
    landmarks = landmarks.reshape(-1, 3)
    base = landmarks[0]
    landmarks -= base
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val > 0:
        landmarks /= max_val

    return landmarks.flatten()
def get_hand_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks_all = []
        handedness = [h.classification[0].label for h in results.multi_handedness]

        # Ensure consistent Left-Right ordering
        hand_pairs = list(zip(handedness, results.multi_hand_landmarks))
        hand_pairs.sort(key=lambda x: 0 if x[0] == "Left" else 1)

        for _, hand_landmarks in hand_pairs:
            for lm in hand_landmarks.landmark:
                landmarks_all.append([lm.x, lm.y, lm.z])

        return normalize_landmarks(np.array(landmarks_all).flatten())
    return None
def register_new_sign():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.7,
                           max_num_hands=2)

    print("\n[INFO] Preparing to capture new sign... Hold your hands steady!")
    time.sleep(2)
    print("[INFO] Capturing for 5 seconds... (Press 'q' to quit early)")

    captured_landmarks = []
    start_time = time.time()

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_hand_landmarks(frame, hands)
        if landmarks is not None:
            captured_landmarks.append(landmarks)
            # Draw landmarks
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Register New Sign", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Registration cancelled by user.\n")
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            return

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if captured_landmarks:
        avg_landmarks = np.mean(captured_landmarks, axis=0)
        meaning = input("Enter the meaning of this new sign: ")
        sign_database[meaning] = avg_landmarks
        save_database()
        print(f"[SUCCESS] New sign '{meaning}' registered successfully!\n")
    else:
        print("[ERROR] No hands detected. Try again.\n")
def translate_sign():
    if not sign_database:
        print("[ERROR] No signs in database. Please register first.\n")
        return

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.7,
                           max_num_hands=2)

    print("\n[INFO] Show a sign to translate. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_hand_landmarks(frame, hands)
        if landmarks is not None:
            best_match = None
            best_distance = float("inf")
            for label, stored_landmarks in sign_database.items():
                min_len = min(len(landmarks), len(stored_landmarks))
                distance = np.linalg.norm(landmarks[:min_len] - stored_landmarks[:min_len])
                if distance < best_distance:
                    best_distance = distance
                    best_match = label

            if best_match:
                cv2.putText(frame, f"Prediction: {best_match}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Translate Sign", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Translation stopped by user.\n")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
def menu():
    while True:
        print("====== SIGN LANGUAGE TRANSLATOR ======")
        print("1. Register New Sign")
        print("2. Translate a Sign")
        print("3. Exit (or press 'q')")

        choice = input("Enter your choice: ")

        if choice == "1":
            register_new_sign()
        elif choice == "2":
            translate_sign()
        elif choice == "3" or choice.lower() == "q":
            print("Exiting... Goodbye!")
            break
        else:
            print("[ERROR] Invalid choice. Please try again.\n")
if __name__ == "__main__":
    load_database()
    menu()
    save_database()