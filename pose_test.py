import cv2
import mediapipe as mp
import math
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

timestamp = 0

HEAD_ON_DESK_THRESHOLD = 0.10
DEEP_SLEEP_TIME = 5.0

head_on_desk_start = None


def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 1

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # Nose
        nose = landmarks[0]

        # Shoulders
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Center of shoulders
        shoulder_x = (
            left_shoulder.x +
            right_shoulder.x
        ) / 2

        shoulder_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2

        # Nose → shoulder center
        shoulder_center = type(
            "Point",
            (),
            {
                "x": shoulder_x,
                "y": shoulder_y
            }
        )()

        nose_to_shoulder = distance(
            nose,
            shoulder_center
        )

        # Draw landmarks
        for landmark in landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Draw shoulder center
        sx = int(shoulder_x * frame.shape[1])
        sy = int(shoulder_y * frame.shape[0])

        cv2.circle(
            frame,
            (sx, sy),
            7,
            (255, 0, 0),
            -1
        )

        # --------------------------------
        # HEAD ON DESK DETECTION
        # --------------------------------

        head_on_desk = (
            nose_to_shoulder <= HEAD_ON_DESK_THRESHOLD
        )

        if head_on_desk:

            # Start timer
            if head_on_desk_start is None:
                head_on_desk_start = time.time()

            elapsed = time.time() - head_on_desk_start

            if elapsed >= DEEP_SLEEP_TIME:

                status = "LIKELY DEEP SLEEPING"

            else:

                status = "HEAD ON DESK"

        else:

            # Reset timer
            head_on_desk_start = None

            elapsed = 0

            status = "NORMAL / LEANING"

        # --------------------------------
        # DISPLAY
        # --------------------------------

        cv2.putText(
            frame,
            f"Nose-Shoulder: {nose_to_shoulder:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        if head_on_desk:

            cv2.putText(
                frame,
                f"Time: {elapsed:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    else:

        # No body → reset
        head_on_desk_start = None

        cv2.putText(
            frame,
            "NO BODY",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Head On Desk Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()