import cv2
import mediapipe as mp
import math
import time

# -----------------------------
# MediaPipe
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

# -----------------------------
# Eye landmarks
# -----------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

def eye_aspect_ratio(eye):
    vertical1 = distance(eye[1], eye[5])
    vertical2 = distance(eye[2], eye[4])
    horizontal = distance(eye[0], eye[3])

    return (vertical1 + vertical2) / (2 * horizontal)

# -----------------------------
# Head landmarks
# -----------------------------

NOSE = 1
FOREHEAD = 10
CHIN = 152

# From your calibration:
# Upright      ≈ 0.50
# Looking down ≈ 0.60
# Desk         ≈ 0.70

HEAD_DOWN_THRESHOLD = 0.57

# -----------------------------
# Detection settings
# -----------------------------

EYE_CLOSED_THRESHOLD = 0.20
SLEEP_TIME = 5.0

eyes_closed_start = None
head_down_closed_start = None

timestamp = 0

# -----------------------------
# Main loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    status = "NO FACE"
    closed_time = 0
    head_down_time = 0

    # IMPORTANT:
    # Everything below happens only when a face exists.

    if result.face_landmarks:

        face = result.face_landmarks[0]

        # -----------------------------
        # Eye detection
        # -----------------------------

        left_eye = [face[i] for i in LEFT_EYE]
        right_eye = [face[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2

        eyes_closed = ear < EYE_CLOSED_THRESHOLD

        # -----------------------------
        # Head position
        # -----------------------------

        nose = face[NOSE]
        forehead = face[FOREHEAD]
        chin = face[CHIN]

        face_height = abs(chin.y - forehead.y)

        if face_height > 0:
            nose_position = (
                nose.y - forehead.y
            ) / face_height
        else:
            nose_position = 0.5

        head_down = nose_position > HEAD_DOWN_THRESHOLD

        # -----------------------------
        # Scenario 1:
        # Eyes closed for 5 seconds
        # -----------------------------

        if eyes_closed:

            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            closed_time = time.time() - eyes_closed_start

        else:

            eyes_closed_start = None
            closed_time = 0

        # -----------------------------
        # Scenario 2:
        # Head down + eyes closed
        # for 5 seconds
        # -----------------------------

        if head_down and eyes_closed:

            if head_down_closed_start is None:
                head_down_closed_start = time.time()

            head_down_time = time.time() - head_down_closed_start

        else:

            head_down_closed_start = None
            head_down_time = 0

        # -----------------------------
        # Final sleeping decision
        # -----------------------------

        if closed_time >= SLEEP_TIME:

            status = "LIKELY SLEEPING"

        elif head_down_time >= SLEEP_TIME:

            status = "LIKELY SLEEPING"

        elif eyes_closed:

            status = "EYES CLOSED"

        elif head_down:

            status = "HEAD DOWN"

        else:

            status = "AWAKE"

        # -----------------------------
        # Display values
        # -----------------------------

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Head: {nose_position:.2f}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eyes closed: {closed_time:.1f}s",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Head down + eyes: {head_down_time:.1f}s",
            (30, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # -----------------------------
        # Draw eye landmarks
        # -----------------------------

        h, w, _ = frame.shape

        for index in LEFT_EYE + RIGHT_EYE:

            x = int(face[index].x * w)
            y = int(face[index].y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

    else:

        # No face = reset everything
        eyes_closed_start = None
        head_down_closed_start = None

    cv2.imshow("Drowsiness Detector", frame)

    # ESC = exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()