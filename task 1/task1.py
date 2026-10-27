import cv2
import numpy as np



"""

Write a program to read a live video stream using VideoCapture and convert each frame to grayscale. 

Bonus: If the stream encounters a corrupted frame, skip it and continue processing the next frames. However, if the total number of corrupted frames exceeds a specific threshold (e.g., 5 frames), break out of the loop completely and terminate the program."

"""
cap = cv2.VideoCapture(0)

# Initialize tracking variables for corrupted frames
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

    # Check if threshold is exceeded
    if corrupted_count > MAX_CORRUPTED_FRAMES:
      print("Error: Too many corrupted frames. Terminating program.")
      break

    # Skip the rest of the loop and try reading the next frame
    continue

  # 1. Convert to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # 2. Convert grayscale back to 3-channel BGR to match shape
  gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

  # 3. Combine both images side-by-side
  combined_frame = np.hstack((frame, gray_3channel))

  # 4. Display the combined result in a single window
  cv2.imshow("Original (Left) vs Grayscale (Right)", combined_frame)

  # Press 'q' to exit manually
  if cv2.waitKey(1) & 0xFF == ord("q"):
    print("User exited manually.")
    break

# Clean up resources
cap.release()
cv2.destroyAllWindows()