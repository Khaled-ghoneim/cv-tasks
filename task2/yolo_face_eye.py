from pathlib import Path

import cv2
from ultralytics import YOLO

# Load model weights relative to this script, regardless of the launch folder.
SCRIPT_DIR = Path(__file__).resolve().parent
face_model = YOLO(str(SCRIPT_DIR / "yolov8n-face.pt"))
pose_model = YOLO(str(SCRIPT_DIR / "yolov8n-pose.pt"))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera permissions.")

print("YOLO Face & Eye Tracking - Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read video frame")
            break

        # Detect faces and estimate pose landmarks for the current frame.
        face_results = face_model.predict(
            frame, imgsz=640, conf=0.4, verbose=False
        )

        # COCO pose keypoints include left- and right-eye landmarks.
        pose_results = pose_model.predict(
            frame, imgsz=640, conf=0.5, verbose=False
        )

        faces = []

        for result in face_results:
            if result.boxes is None:
                continue

            for box in result.boxes.xyxy.cpu().numpy():
                # Convert model coordinates to integers and clamp them to the
                # frame so later crops and drawings stay within its bounds.
                x1, y1, x2, y2 = map(int, box[:4])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                faces.append((x1, y1, x2, y2))

                # Save each valid face for filtering eye points, then draw it.
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 255), 2
                )
                cv2.putText(
                    frame, "Face",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2
                )

        for result in pose_results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()

            if result.keypoints.conf is not None:
                confidences = result.keypoints.conf.cpu().numpy()
            else:
                confidences = None

            for person_idx, points in enumerate(keypoints):
                # Pose landmark indices 1 and 2 represent the two eyes.
                for idx, color, label in [
                    (1, (255, 0, 0), "Left Eye"),
                    (2, (0, 0, 255), "Right Eye")
                ]:
                    if confidences is not None:
                        # Discard uncertain eye predictions.
                        if confidences[person_idx][idx] < 0.4:
                            continue

                    ex, ey = map(int, points[idx])

                    # Ignore eye points that do not fall inside a face box.
                    face = next(
                        (
                            (x1, y1, x2, y2)
                            for x1, y1, x2, y2 in faces
                            if x1 <= ex <= x2 and y1 <= ey <= y2
                        ),
                        None
                    )

                    if face is None:
                        continue

                    x1, y1, x2, y2 = face
                    face_width = x2 - x1

                    # Pose gives eye points, not boxes; size a display box in
                    # proportion to the detected face.
                    box_w = max(8, int(face_width * 0.10))
                    box_h = max(6, int(face_width * 0.06))

                    cv2.rectangle(
                        frame,
                        (max(x1, ex - box_w // 2),
                         max(y1, ey - box_h // 2)),
                        (min(x2, ex + box_w // 2),
                         min(y2, ey + box_h // 2)),
                        color, 2
                    )
                    cv2.circle(frame, (ex, ey), 3, color, -1)

        cv2.imshow("YOLO Face & Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
