"""Real-time webcam air writing with MediaPipe hand tracking.

Move the cursor with your index fingertip and pinch thumb to index to draw.
Release the pinch to lift the pen. C clears; R recognizes; Q or Escape quits.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = Path(__file__).with_name("letter_svm.yml")
HAND_TASK_PATH = Path(__file__).with_name("hand_landmarker.task")
HAND_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CANVAS_SIZE = 640
SMOOTHING = 0.35
PINCH_START_RATIO = 0.45
PINCH_RELEASE_RATIO = 0.58
LETTER_HOG = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)


def make_recognition_image(canvas: np.ndarray) -> np.ndarray:
    """Crop ink, preserve its aspect ratio, and center it in a 28x28 image."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 20)
    if len(xs) == 0:
        raise ValueError("Write a letter before asking for recognition.")

    pad = 12
    x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad + 1)
    crop = gray[y0:y1, x0:x1]
    # EMNIST images are white ink on black; the drawing canvas is already that way.
    scale = 20.0 / max(crop.shape)
    resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                max(1, round(crop.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    result = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - resized.shape[0]) // 2, (28 - resized.shape[1]) // 2
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result


def recognize_text(canvas: np.ndarray, classifier) -> str:
    """Segment separated handwriting into components and classify A-Z/space.

    This supports block or air-written letters with visible horizontal gaps.
    Connected cursive writing needs a sequence model and is not handled here.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ink = (gray > 20).astype(np.uint8)
    if not np.any(ink):
        raise ValueError("Write a letter or word before pressing R.")

    # Join small gaps inside a letter while retaining larger gaps between letters.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= 18 and h >= 8 and w >= 3:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: box[0])
    if not boxes:
        raise ValueError("No clear letter shapes found. Write larger, separated letters.")

    heights = [h for _, _, _, h in boxes]
    typical_height = float(np.median(heights))
    parts: list[str] = []
    previous_right = None
    for x, y, w, h in boxes:
        if previous_right is not None:
            gap = x - previous_right
            if gap > max(18, typical_height * 0.55):
                parts.append(" ")
        # Use original strokes for classification; morphology is only for boxes.
        component = np.zeros_like(canvas)
        component[y:y + h, x:x + w] = canvas[y:y + h, x:x + w]
        normalized = make_recognition_image(component)
        sample = LETTER_HOG.compute(normalized).reshape(1, -1).astype(np.float32)
        _, response = classifier.predict(sample)
        class_id = int(response[0, 0])
        parts.append(chr(ord("A") + class_id - 1) if 1 <= class_id <= 26 else "?")
        previous_right = x + w
    return "".join(parts)


def _box_for_strokes(strokes: list[list[tuple[int, int]]]) -> tuple[int, int, int, int]:
    xs = [p[0] for stroke in strokes for p in stroke]
    ys = [p[1] for stroke in strokes for p in stroke]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def recognize_strokes(strokes: list[list[tuple[int, int]]], canvas: np.ndarray, classifier) -> str:
    """Group pen strokes into characters using spatial overlap and classify them.

    This keeps multi-stroke block letters together. A substantially larger gap
    between character groups becomes a word space. It is still a baseline for
    separated block writing, not cursive handwriting OCR.
    """
    valid = [stroke for stroke in strokes if len(stroke) >= 2]
    if not valid:
        raise ValueError("Write a word with strokes long enough to recognize.")

    # Process strokes in the order written. A stroke joins the closest recent
    # character whose horizontal span overlaps and whose vertical distance is
    # plausible for a crossbar or a dot.
    groups: list[list[list[tuple[int, int]]]] = []
    for stroke in valid:
        sx0, sy0, sx1, sy1 = _box_for_strokes([stroke])
        best_group = None
        best_distance = float("inf")
        for group in groups[-4:]:
            gx0, gy0, gx1, gy1 = _box_for_strokes(group)
            overlap = max(0, min(sx1, gx1) - max(sx0, gx0))
            min_width = max(1, min(sx1 - sx0, gx1 - gx0))
            horizontal_near = sx0 <= gx1 + 20 and gx0 <= sx1 + 20
            vertical_gap = max(0, max(gy0 - sy1, sy0 - gy1))
            max_height = max(gy1 - gy0, sy1 - sy0)
            if horizontal_near and (overlap / min_width >= 0.18 or vertical_gap < max_height * 0.42):
                distance = abs((sx0 + sx1) / 2 - (gx0 + gx1) / 2) + vertical_gap
                if distance < best_distance:
                    best_group, best_distance = group, distance
        if best_group is None:
            groups.append([stroke])
        else:
            best_group.append(stroke)

    groups.sort(key=lambda group: _box_for_strokes(group)[0])
    boxes = [_box_for_strokes(group) for group in groups]
    typical_height = float(np.median([y1 - y0 for _, y0, _, y1 in boxes]))
    result = []
    previous_right = None
    for group, (x0, y0, x1, y1) in zip(groups, boxes):
        if previous_right is not None and x0 - previous_right > max(28, typical_height * 0.75):
            result.append(" ")
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        for stroke in group:
            for a, b in zip(stroke, stroke[1:]):
                cv2.line(mask, a, b, 255, 8, cv2.LINE_AA)
            if len(stroke) == 1:
                cv2.circle(mask, stroke[0], 4, 255, -1, cv2.LINE_AA)
        # Include a small crop margin so strokes near the grouping boundary survive.
        pad = 6
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(mask.shape[1], x1 + pad), min(mask.shape[0], y1 + pad)
        group_canvas = cv2.cvtColor(mask[cy0:cy1, cx0:cx1], cv2.COLOR_GRAY2BGR)
        normalized = make_recognition_image(group_canvas)
        feature = LETTER_HOG.compute(normalized).reshape(1, -1).astype(np.float32)
        _, response = classifier.predict(feature)
        label = int(response[0, 0])
        result.append(chr(ord("A") + label - 1) if 1 <= label <= 26 else "?")
        previous_right = x1
    return "".join(result)


def draw_hand_skeleton(frame: np.ndarray, landmarks, width: int, height: int) -> None:
    """Draw MediaPipe's hand connections and landmarks as a readable overlay."""
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    points = [(int(np.clip(p.x, 0, 1) * width), int(np.clip(p.y, 0, 1) * height))
              for p in landmarks]
    for a, b in connections:
        cv2.line(frame, points[a], points[b], (80, 220, 120), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        color = (0, 80, 255) if index in (4, 8) else (255, 210, 60)
        cv2.circle(frame, point, 4 if index in (4, 8) else 3, color, -1, cv2.LINE_AA)


def blur_background_around_hand(frame: np.ndarray, landmarks) -> np.ndarray:
    """Blur preview background efficiently at reduced resolution around the hand."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    small_width, small_height = max(1, int(width * scale)), max(1, int(height * scale))
    small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
    points = np.asarray([(int(p.x * small_width), int(p.y * small_height)) for p in landmarks], dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((small_height, small_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    radius = max(8, int(min(small_width, small_height) * 0.045))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, radius / 2))
    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=7, sigmaY=7)
    alpha = mask.astype(np.float32)[:, :, None] / 255.0
    preview = (small.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return cv2.resize(preview, (width, height), interpolation=cv2.INTER_LINEAR)


def create_hand_tracker(mp):
    """Return a tracker adapter for either legacy Solutions or MediaPipe Tasks."""
    if hasattr(mp, "solutions"):
        api = mp.solutions.hands
        tracker = api.Hands(max_num_hands=1, model_complexity=0,
                            min_detection_confidence=0.65,
                            min_tracking_confidence=0.6)
        return "solutions", tracker, api, mp.solutions.drawing_utils

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("This MediaPipe package has neither Solutions nor Tasks Hands support.") from exc

    if not HAND_TASK_PATH.exists():
        print("Downloading the MediaPipe hand landmark model (about 8 MB)...")
        try:
            urllib.request.urlretrieve(HAND_TASK_URL, HAND_TASK_PATH)
        except Exception as exc:
            raise RuntimeError(f"Could not download hand model: {exc}\nDownload it manually from {HAND_TASK_URL} and save it as {HAND_TASK_PATH}") from exc

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return "tasks", vision.HandLandmarker.create_from_options(options), vision, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Write in the air by pinching thumb and index finger.")
    parser.add_argument("--camera", type=int, default=0, help="webcam device index (default: 0)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="trained OpenCV SVM model")
    args = parser.parse_args()

    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe is missing. Install dependencies with: python -m pip install -r requirements.txt")
        return 2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Check camera permissions or pass --camera N.")
        return 2

    try:
        tracker_kind, hands, hands_api, drawing = create_hand_tracker(mp)
    except RuntimeError as exc:
        cap.release()
        print(exc)
        return 2
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    strokes: list[list[tuple[int, int]]] = []
    current_stroke: list[tuple[int, int]] | None = None
    smoothed: tuple[float, float] | None = None
    pinch_drawing = False
    classifier = cv2.ml.SVM_load(str(args.model)) if args.model.exists() else None
    prediction = "No model: drawing only"
    print("Move your index fingertip to position the cursor. Pinch thumb and index to write; release to lift the pen.")

    frame_timestamp_ms = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame could not be read.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if tracker_kind == "solutions":
                rgb.flags.writeable = False
                result = hands.process(rgb)
                rgb.flags.writeable = True
                detected_hands = result.multi_hand_landmarks or []
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hands.detect_for_video(mp_image, frame_timestamp_ms)
                frame_timestamp_ms += max(1, int(1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30)))
                detected_hands = result.hand_landmarks or []

            writing = False
            point = None
            pinch_ratio = None
            current_landmarks = None
            if detected_hands:
                hand = detected_hands[0]
                lm = hand.landmark if tracker_kind == "solutions" else hand
                current_landmarks = lm
                def distance(a, b):
                    # Include depth so hand rotation is less likely to look like a pinch.
                    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))

                # Normalize thumb-index separation by palm width to adapt to scale.
                palm_width = max(distance(lm[5], lm[17]), 1e-6)
                pinch_ratio = distance(lm[4], lm[8]) / palm_width
                if pinch_drawing:
                    pinch_drawing = pinch_ratio < PINCH_RELEASE_RATIO
                else:
                    pinch_drawing = pinch_ratio < PINCH_START_RATIO
                writing = pinch_drawing
                point = (int(np.clip(lm[8].x, 0, 1) * (CANVAS_SIZE - 1)),
                         int(np.clip(lm[8].y, 0, 1) * (CANVAS_SIZE - 1)))

            if current_landmarks is not None and width * height <= 1280 * 720:
                frame = blur_background_around_hand(frame, current_landmarks)
            if current_landmarks is not None:
                draw_hand_skeleton(frame, current_landmarks, width, height)

            board_cursor = None
            if point is not None:
                if smoothed is None:
                    smoothed = (float(point[0]), float(point[1]))
                else:
                    smoothed = (SMOOTHING * point[0] + (1 - SMOOTHING) * smoothed[0],
                                SMOOTHING * point[1] + (1 - SMOOTHING) * smoothed[1])
                p = (int(smoothed[0]), int(smoothed[1]))
                board_cursor = p
                if writing:
                    if current_stroke is None:
                        current_stroke = [p]
                        strokes.append(current_stroke)
                    elif p != current_stroke[-1]:
                        current_stroke.append(p)
                    cv2.circle(frame, p, 9, (0, 0, 255), -1, cv2.LINE_AA)
                else:
                    current_stroke = None
                    cv2.circle(frame, p, 9, (0, 220, 0), -1, cv2.LINE_AA)
                    if pinch_ratio is not None:
                        cv2.putText(frame, f"Pinch ratio: {pinch_ratio:.2f}", (20, 108),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
            else:
                current_stroke = None
                smoothed = None
                pinch_drawing = False

            for stroke in strokes:
                for a, b in zip(stroke, stroke[1:]):
                    cv2.line(canvas, a, b, (255, 255, 255), 8, cv2.LINE_AA)
            # Red start markers stay on the camera preview; recognition canvas stays clean.
            for stroke in strokes:
                if stroke:
                    marker = (int(stroke[0][0] * width / CANVAS_SIZE),
                              int(stroke[0][1] * height / CANVAS_SIZE))
                    cv2.circle(frame, marker, 6, (0, 0, 255), -1, cv2.LINE_AA)

            # Draw a live cursor after the ink so its current position is always
            # visible on the board. It is only a preview: it never enters the
            # recognition canvas. Red stroke-start hints are shown separately.
            if board_cursor is not None:
                cursor_color = (0, 0, 255) if writing else (0, 220, 0)
                board_preview = canvas.copy()
                cv2.circle(board_preview, board_cursor, 8, cursor_color, -1, cv2.LINE_AA)
                camera_cursor = (int(board_cursor[0] * width / CANVAS_SIZE),
                                 int(board_cursor[1] * height / CANVAS_SIZE))
                cv2.circle(frame, camera_cursor, 8, cursor_color, -1, cv2.LINE_AA)
            else:
                board_preview = canvas

            status = "WRITING - release pinch to lift" if writing else "PEN UP - pinch thumb + index to write"
            cv2.putText(frame, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if writing else (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "C: clear  R: recognize  Q: quit", (20, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, prediction, (20, 72), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 220, 0), 2, cv2.LINE_AA)

            board_width = min(640, max(360, width // 2))
            board = cv2.resize(board_preview, (board_width, board_width), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Air Writing AI - camera", frame)
            cv2.imshow("Air Writing AI - canvas", board)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                strokes.clear()
                current_stroke = None
                prediction = "Canvas cleared"
                canvas.fill(0)
            elif key == ord("r"):
                if classifier is None:
                    prediction = "Train A-Z model first (see README)"
                else:
                    try:
                        prediction = recognize_strokes(strokes, canvas, classifier)
                    except ValueError as exc:
                        prediction = str(exc)
                    except cv2.error as exc:
                        prediction = "Model incompatible; retrain with train_classifier.py"
                        print(f"Recognition error: {exc}")
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
