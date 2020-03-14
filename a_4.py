# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 00:58:23 2020

@author: K555D
"""

import numpy as np
import cv2


# Load images:
img_1 = cv2.imread('1.jpg', 0)
img_2 = cv2.imread('2.jpg', 0)


def edgeDetection(img, title):
    
# =============================================================================
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# =============================================================================
    
    # Canny 
    canny_img = cv2.Canny(img, 100, 200)
    canny_img_blur = cv2.GaussianBlur(canny_img, (3,3), 0)
    
    cv2.imwrite('Canny_blur_'+title+'.jpg', canny_img_blur)
    cv2.imwrite('Canny_'+title+'.jpg', canny_img)
    
    # LoG
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    _, laplacian1 = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite('LoG_thresh_'+title+'.jpg', laplacian1)
    cv2.imwrite('LoG_'+title+'.jpg', laplacian)
    
    # Sobel
    sobelx = cv2.Sobel(img_blur,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img_blur,cv2.CV_64F,0,1,ksize=3)
    sobel = np.uint8(np.sqrt(sobelx**2+sobely**2))

    cv2.imwrite('Sobelx_'+title+'.jpg', sobelx)
    cv2.imwrite('Sobely_'+title+'.jpg', sobely)
    cv2.imwrite('Sobel_'+title+'.jpg', sobel)
    
edgeDetection(img_2, 'img_2')
edgeDetection(img_1, 'img_1')

cv2.waitKey(0)
cv2.destroyAllWindows()