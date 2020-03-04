# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:15:21 2020

@author: K555D
"""

import cv2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('recorded.avi', fourcc, 30.0, (640, 480))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
save = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        continue

    if (save == 1):
        out.write(frame)

    # Display
    cv2.imshow('Live Stream', frame)

    k = cv2.waitKey(1)
    if (k == ord('s')):
        save = 1
        print("Recording...")
    elif (k == ord('e')):
        print("Stopped!")
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()