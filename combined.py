import cv2
import mediapipe as mp
import math
import time


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


# ============================================================
# FACE LANDMARKER
# ============================================================

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = FaceLandmarker.create_from_options(
    face_options
)


# ============================================================
# POSE LANDMARKER
# ============================================================

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

timestamp = 0


# ============================================================
# DETECTOR 1 SETTINGS
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

NOSE = 1
FOREHEAD = 10
CHIN = 152

EYE_CLOSED_THRESHOLD = 0.20
HEAD_DOWN_THRESHOLD = 0.57

SLEEP_TIME = 5.0

eyes_closed_start = None
head_down_closed_start = None


# ============================================================
# DETECTOR 2 SETTINGS
# ============================================================

HEAD_ON_DESK_THRESHOLD = 0.10
DEEP_SLEEP_TIME = 5.0

head_on_desk_start = None


# ============================================================
# DETECTOR 3 SETTINGS
# ============================================================

YAWN_THRESHOLD = 1.0
YAWN_TIME = 0.7

yawn_start = None
yawn_detected = False


# ============================================================
# HELPER
# ============================================================

def distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def calculate_ear(landmarks, eye_indices):

    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    return (
        vertical1 + vertical2
    ) / (2 * horizontal)


# ============================================================
# MAIN LOOP
# ============================================================

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


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_result = face_landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # POSE DETECTION
    # ========================================================

    pose_result = pose_landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    timestamp += 1


    # ========================================================
    # DETECTOR 1 + 3
    # FACE BASED
    # ========================================================

    if face_result.face_landmarks:

        landmarks = face_result.face_landmarks[0]


        # ----------------------------------------------------
        # EYE DETECTION
        # ----------------------------------------------------

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear + right_ear
        ) / 2

        eyes_closed = (
            ear < EYE_CLOSED_THRESHOLD
        )


        # ----------------------------------------------------
        # HEAD DOWN
        # ----------------------------------------------------

        nose = landmarks[NOSE]
        forehead = landmarks[FOREHEAD]
        chin = landmarks[CHIN]

        face_height = abs(
            chin.y - forehead.y
        )

        if face_height > 0:

            nose_position = (
                nose.y - forehead.y
            ) / face_height

            head_down = (
                nose_position >
                HEAD_DOWN_THRESHOLD
            )

        else:

            head_down = False


        # ----------------------------------------------------
        # EYES CLOSED TIMER
        # ----------------------------------------------------

        if eyes_closed:

            if eyes_closed_start is None:
                eyes_closed_start = time.time()

        else:

            eyes_closed_start = None


        # ----------------------------------------------------
        # HEAD DOWN + EYES CLOSED TIMER
        # ----------------------------------------------------

        if head_down and eyes_closed:

            if head_down_closed_start is None:
                head_down_closed_start = time.time()

        else:

            head_down_closed_start = None


        # ----------------------------------------------------
        # SLEEPING RESULT
        # ----------------------------------------------------

        eyes_closed_time = 0
        head_down_closed_time = 0

        if eyes_closed_start is not None:

            eyes_closed_time = (
                time.time() -
                eyes_closed_start
            )

        if head_down_closed_start is not None:

            head_down_closed_time = (
                time.time() -
                head_down_closed_start
            )


        likely_sleeping = (
            eyes_closed_time >= SLEEP_TIME
            or
            head_down_closed_time >= SLEEP_TIME
        )


        # ====================================================
        # YAWN DETECTOR
        # ====================================================

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

        mouth_ratio = (
            mouth_height /
            mouth_width
        )

        mouth_wide_open = (
            mouth_ratio >= YAWN_THRESHOLD
        )


        if mouth_wide_open:

            if yawn_start is None:
                yawn_start = time.time()

            yawn_elapsed = (
                time.time() -
                yawn_start
            )

            if yawn_elapsed >= YAWN_TIME:
                yawn_detected = True

        else:

            yawn_start = None
            yawn_detected = False


        # ====================================================
        # DRAW FACE LANDMARKS
        # ====================================================

        for landmark in landmarks:

            x = int(
                landmark.x *
                frame.shape[1]
            )

            y = int(
                landmark.y *
                frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )


    else:

        # No face → reset face timers

        eyes_closed_start = None
        head_down_closed_start = None

        yawn_start = None
        yawn_detected = False

        ear = 0
        mouth_ratio = 0

        eyes_closed = False
        head_down = False

        likely_sleeping = False


    # ========================================================
    # DETECTOR 2
    # BODY / HEAD ON DESK
    # ========================================================

    if pose_result.pose_landmarks:

        pose = pose_result.pose_landmarks[0]

        nose = pose[0]

        left_shoulder = pose[11]
        right_shoulder = pose[12]

        shoulder_x = (
            left_shoulder.x +
            right_shoulder.x
        ) / 2

        shoulder_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2


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


        head_on_desk = (
            nose_to_shoulder <=
            HEAD_ON_DESK_THRESHOLD
        )


        if head_on_desk:

            if head_on_desk_start is None:
                head_on_desk_start = time.time()

            head_on_desk_time = (
                time.time() -
                head_on_desk_start
            )

        else:

            head_on_desk_start = None
            head_on_desk_time = 0


        likely_deep_sleeping = (
            head_on_desk_time >=
            DEEP_SLEEP_TIME
        )


        # Draw pose landmarks

        for landmark in pose:

            x = int(
                landmark.x *
                frame.shape[1]
            )

            y = int(
                landmark.y *
                frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )


    else:

        head_on_desk_start = None
        head_on_desk_time = 0

        nose_to_shoulder = 0

        head_on_desk = False
        likely_deep_sleeping = False


    # ========================================================
    # DISPLAY
    # ========================================================

    y = 35


    # Detector 1

    if likely_sleeping:

        text = "LIKELY SLEEPING"

    elif eyes_closed:

        text = f"EYES CLOSED ({eyes_closed_time:.1f}s)"

    elif head_down:

        text = "HEAD DOWN"

    else:

        text = "AWAKE"

    cv2.putText(
        frame,
        f"FACE: {text}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 35


    # Detector 2

    if likely_deep_sleeping:

        text = "LIKELY DEEP SLEEPING"

    elif head_on_desk:

        text = f"HEAD ON DESK ({head_on_desk_time:.1f}s)"

    else:

        text = "BODY NORMAL"

    cv2.putText(
        frame,
        f"BODY: {text}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    y += 35


    # Detector 3

    if yawn_detected:

        text = "YAWNING - SLEEPY"

    elif mouth_ratio >= YAWN_THRESHOLD:

        text = "MOUTH OPEN"

    else:

        text = "NO YAWN"

    cv2.putText(
        frame,
        f"YAWN: {text}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Student Sleep Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

face_landmarker.close()
pose_landmarker.close()