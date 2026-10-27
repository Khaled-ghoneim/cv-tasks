from pathlib import Path

import cv2

# Resolve cascade files relative to this script so it works from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent
face_cascade = cv2.CascadeClassifier(
    str(SCRIPT_DIR / "haarcascade_frontalface_default.xml")
)
eye_cascade = cv2.CascadeClassifier(str(SCRIPT_DIR / "haarcascade_eye.xml"))

# Safety check to ensure classifiers loaded successfully
if face_cascade.empty() or eye_cascade.empty():
  raise RuntimeError("Error loading Haar cascade XML files. Check OpenCV installation.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera permissions.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read video frame")
        break

    # Haar classifiers process grayscale frames.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # Search for eyes only inside this face's grayscale crop.
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Eye coordinates are relative to the face crop, so draw on it.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 3)

    cv2.imshow("Face and Eye Detection - Haar Cascade", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the preview window.
cap.release()
cv2.destroyAllWindows()
