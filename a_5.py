import cv2 
import numpy as np 

# Reading the input image 
img = cv2.imread('limbo.png') 

# Taking matrices of size 3,5,7 as the kernels:
 
kernel1 = np.ones((3,3), np.uint8) 
kernel2 = np.ones((5,5), np.uint8) 
kernel3 = np.ones((7,7), np.uint8) 

# Image erosion:
img_eroded1 = cv2.erode(img, kernel1, iterations=1) 
img_eroded2 = cv2.erode(img, kernel2, iterations=1) 
img_eroded3 = cv2.erode(img, kernel3, iterations=1) 

# Image dilation
img_dilated1 = cv2.dilate(img, kernel1, iterations=1)
img_dilated2 = cv2.dilate(img, kernel2, iterations=1)
img_dilated3 = cv2.dilate(img, kernel3, iterations=1)

# Closing is a dilation followed by erosion
closing = cv2.erode(img_dilated3, kernel3, iterations=1) 

# Opening is a erosion followed by dilation
opening = cv2.dilate(img_eroded3, kernel3, iterations=1)



cv2.imshow('Original', img) 
cv2.imshow('Erosion1', img_eroded1)
cv2.imshow('Erosion2', img_eroded2) 
cv2.imshow('Erosion3', img_eroded3) 
 
cv2.imshow('Dilation1', img_dilated1) 
cv2.imshow('Dilation2', img_dilated2) 
cv2.imshow('Dilation3', img_dilated3) 
  
cv2.imshow('Opening',opening)
cv2.imshow('Closing',closing)


k = cv2.waitKey(0)

if k == ord('s'):
    cv2.imwrite('img_eroded1.jpg',img_eroded1)
    cv2.imwrite('img_eroded2.jpg',img_eroded2)
    cv2.imwrite('img_eroded3.jpg',img_eroded3)
    cv2.imwrite('img_dilated1.jpg',img_dilated1)
    cv2.imwrite('img_dilated2.jpg',img_dilated2)
    cv2.imwrite('img_dilated3.jpg',img_dilated3)
    cv2.imwrite('img_closing.jpg',closing)
    cv2.imwrite('img_opening.jpg',opening)
    
    cv2.destroyAllWindows()
elif k == ord('e'):         # wait for 'e' key to exit
    cv2.destroyAllWindows()