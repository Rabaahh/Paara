import cv2
import mediapipe as mp
import math
import time

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

cap = cv2.VideoCapture(0)

timestamp = 0

YAWN_THRESHOLD = 1.0
YAWN_TIME = 0.7

yawn_start = None
yawn_detected = False


def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 1

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        mouth_height = distance(
            upper_lip,
            lower_lip
        )

        mouth_width = distance(
            left_mouth,
            right_mouth
        )

        mouth_ratio = mouth_height / mouth_width

        # Draw mouth landmarks
        for index in [13, 14, 61, 291]:

            landmark = landmarks[index]

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # -------------------------
        # YAWN DETECTION
        # -------------------------

        mouth_wide_open = (
            mouth_ratio >= YAWN_THRESHOLD
        )

        if mouth_wide_open:

            # Start timer
            if yawn_start is None:
                yawn_start = time.time()

            elapsed = time.time() - yawn_start

            # Detect only once per mouth opening
            if elapsed >= YAWN_TIME:
                yawn_detected = True

        else:

            # Mouth closed again
            yawn_start = None
            yawn_detected = False

        # -------------------------
        # DISPLAY
        # -------------------------

        cv2.putText(
            frame,
            f"Mouth ratio: {mouth_ratio:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if yawn_detected:

            cv2.putText(
                frame,
                "YAWNING - SLEEPY",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        elif mouth_wide_open:

            cv2.putText(
                frame,
                "MOUTH OPEN",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NORMAL",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    else:

        # No face → reset
        yawn_start = None
        yawn_detected = False

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Yawn Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()