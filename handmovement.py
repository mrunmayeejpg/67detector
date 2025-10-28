import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

img_to_show = cv2.imread(r'C:\Users\This pc\Downloads\meme.jpg')
if img_to_show is None:
    print("Failed to load image! Check your path.")
    img_to_show = 255 * np.ones((200, 200, 3), dtype=np.uint8)
    cv2.putText(img_to_show, 'Image not found', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cap = cv2.VideoCapture(0)

prev_left_y = None
prev_right_y = None

state = 0
cycle_count = 0
image_shown = False
movement_threshold = 0.02

def is_palm_facing_up(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    mcp = hand_landmarks.landmark[9]
    return wrist.y > mcp.y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_y = None
    right_y = None
    left_palm_up = False
    right_palm_up = False

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist_y = hand_landmarks.landmark[0].y

            label = handedness.classification[0].label
            palm_up = is_palm_facing_up(hand_landmarks)

            if label == 'Left':
                left_y = wrist_y
                left_palm_up = palm_up
            elif label == 'Right':
                right_y = wrist_y
                right_palm_up = palm_up

    if (
        left_y is not None and right_y is not None
        and prev_left_y is not None and prev_right_y is not None
        and left_palm_up and right_palm_up
    ):
        left_move = prev_left_y - left_y  # positive = left hand moved up
        right_move = prev_right_y - right_y  # positive = right hand moved up

        if state == 0:
            # Left hand goes UP (left_move > threshold) and Right hand goes DOWN (right_move < -threshold)
            if left_move > movement_threshold and right_move < -movement_threshold:
                print("State 0 completed: Left UP, Right DOWN")
                state = 1
        elif state == 1:
            # Left hand goes DOWN (left_move < -threshold) and Right hand goes UP (right_move > threshold)
            if left_move < -movement_threshold and right_move > movement_threshold:
                cycle_count += 1
                print(f"Cycle {cycle_count} completed!")
                state = 0
                image_shown = False  # Reset image show flag to show image again

        if cycle_count >= 1 and not image_shown:
            cv2.imshow('Detected Image', img_to_show)
            image_shown = True

    else:
        # If hands not detected or palms not up, do nothing or keep image shown
        pass

    prev_left_y = left_y
    prev_right_y = right_y

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
