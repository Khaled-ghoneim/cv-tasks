import cv2 
import matplotlib.pyplot as plt
import numpy as np

"""
Write a program to read a live video stream using VideoCapture and implement real-time Face Detection and Eye Detection using Haar Cascades.
Your script must process the video frame-by-frame, detect the face region,
and locate the eyes inside that specific face region using OpenCV's built-in classifiers.

Bonus:

     Search for and implement a modern, advanced face and eye detection model that performs better than Haar Cascades.
"""
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

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

    
  
    imgg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(imgg)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        
        roi_gray = imgg[y:y+h, x:x+w] # Crop face region from grayscale image (for eye detection)
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
       

        for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255, 255, 0), 3)

   
    cv2.imshow("Face and Eye Detection - Haar Cascade", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
