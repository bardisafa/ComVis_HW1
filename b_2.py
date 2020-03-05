# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:47:49 2020

@author: K555D
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:15:21 2020

@author: K555D
"""

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
cap1 = cv2.VideoCapture('video.mp4')
# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# =============================================================================
medianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
# =============================================================================
# Get current width of frame
width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
# Get current height of frame
height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('Output.avi', fourcc, 30.0, (int(width), int(height)))

while True:
    ret, frame1 = cap1.read()

    if ret == True:
        # =============================================================================
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # =============================================================================
        difference = cv2.absdiff(medianFrame, frame1)
        # =============================================================================
        _, difference = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
        difference = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)
        # =============================================================================
        out.write(difference)

        cv2.imshow("Differnce", difference)

        k = cv2.waitKey(30)

        if k == ord('e'):
            break

    else:
        break

cap.release()
cap1.release()
out.release()
cv2.destroyAllWindows()
