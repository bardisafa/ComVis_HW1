# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 02:54:11 2020

@author: K555D
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('4.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
_,img_gray = cv2.threshold(img_gray,20,255,cv2.THRESH_BINARY)


template = img_gray[311:332, 82:116]

w, h = template.shape[::-1]


threshold = 0.75
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 4)

cv2.imshow('qq.jpg', img_rgb)
cv2.imshow('TemplateMatching.jpg', res)
cv2.imwrite('TemplateMatching.png', img_rgb)
cv2.imwrite('template.png', template)
cv2.imwrite('img_gray.png', img_gray)
cv2.waitKey(0)

