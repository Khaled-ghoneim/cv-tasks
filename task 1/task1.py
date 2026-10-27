import cv2
import numpy as np

# Read from the default webcam. Change this index if the desired camera is
# connected under a different device number.
cap = cv2.VideoCapture(0)

# Stop after repeated failed reads so a disconnected camera cannot leave the
# preview loop running forever.
corrupted_count = 0
MAX_CORRUPTED_FRAMES = 5

while True:
  ret, frame = cap.read()

  # Check if the frame is corrupted or failed to read
  if not ret or frame is None:
    corrupted_count += 1
    print(
        f"Warning: Corrupted frame encountered! Count:"
        f" {corrupted_count}/{MAX_CORRUPTED_FRAMES}"
    )

    # Allow up to five failed reads; the sixth ends the program.
    if corrupted_count > MAX_CORRUPTED_FRAMES:
      print("Error: Too many corrupted frames. Terminating program.")
      break

    # Skip the rest of the loop and try reading the next frame
    continue

  # 1. Convert to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Convert grayscale back to three channels so both images have equal shapes.
  gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

  # 3. Combine both images side-by-side
  combined_frame = np.hstack((frame, gray_3channel))

  # 4. Display the combined result in a single window
  cv2.imshow("Original (Left) vs Grayscale (Right)", combined_frame)

  # Press 'q' to exit manually
  if cv2.waitKey(1) & 0xFF == ord("q"):
    print("User exited manually.")
    break

# Release the webcam and close any OpenCV windows.
cap.release()
cv2.destroyAllWindows()
