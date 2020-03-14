# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:04:12 2020

@author: K555D
"""

import numpy as np
import cv2
import math

img = cv2.imread('2.jpg')

# resize image:
dim = (400, 300)
img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# =============================================================================
# our LPF == gaussian filter
# =============================================================================
# 2D gaussian function:
def gaussian(x, y, sigma):
    return (1/(2*math.pi*sigma**2))*math.exp((x**2 + y**2)/(2*sigma**2))

# 2D gaussian kernel:
sigma = 1
size = 1 + 4*sigma
gauss_kernel = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        gauss_kernel[i, j] = gaussian(i-2*sigma, j-2*sigma, sigma)
gauss_kernel = gauss_kernel/gauss_kernel.sum()   
     
# =============================================================================
# define 2D convolution
# =============================================================================
def convolve2D(mat, kernel):
    
    pad_size_h = int((kernel.shape[0]-1)/2)
    pad_size_w = int((kernel.shape[1]-1)/2)
    
    padded_mat = cv2.copyMakeBorder(mat, pad_size_h, pad_size_h, pad_size_w, pad_size_w, cv2.BORDER_CONSTANT)
    
    height, width = mat.shape
    output_image = np.zeros((height, width))        # make a matrix of size  of original image.
    for h in range(height):
        for w in range(width):
            for h_k in range(kernel.shape[0]):
                for w_k in range(kernel.shape[1]):
                    output_image[h][w] =  output_image[h][w]\
                    + padded_mat[h + h_k][w + w_k] * kernel[h_k][w_k]
    return output_image

# =============================================================================
# split image to b,g,r channels and apply kernel to each of them separately
# =============================================================================
b, g, r = cv2.split(img_resized)
b_lpf = convolve2D(b, gauss_kernel)
g_lpf = convolve2D(g, gauss_kernel)
r_lpf = convolve2D(r, gauss_kernel)

# convert np.ndarray to uint8
b_lpf = b_lpf.astype(np.uint8)
g_lpf = g_lpf.astype(np.uint8)
r_lpf = r_lpf.astype(np.uint8)

# low-passed image:
img_lpf = cv2.merge([b_lpf, g_lpf, r_lpf])

cv2.imshow('Img_LPF', img_lpf)
cv2.imwrite('Img_LPF.jpg', img_lpf)

# =============================================================================
# kernels for horizontal and vertical edge detection
# =============================================================================

prewitt_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])     # Vertical    
prewitt_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])     # Horizontal

# =============================================================================
# vertical edge detection
# =============================================================================
# Averaging between 3 channels:
b_v, g_v, r_v = cv2.split(img_lpf)
channel_avg_v = (1/3)*(b_v + g_v + r_v)

# absolute gradient matrix:
abs_grad_v = np.abs(convolve2D(channel_avg_v, prewitt_v))

# convert np.ndarray to uint8
abs_grad_v = abs_grad_v.astype(np.uint8)

# Threshold
_, abs_grad_v = cv2.threshold(abs_grad_v, 30, 255,cv2.THRESH_BINARY)
# =============================================================================
cv2.imshow('img_LP_v', abs_grad_v)
cv2.imwrite('img_LP_v.jpg', abs_grad_v)

# =============================================================================
# horizontal edge detection
# =============================================================================
b_h, g_h, r_h = cv2.split(img_lpf)

# Averaging between 3 channels:
channel_avg_h = (1/3)*(b_h + g_h + r_h)


abs_grad_h = np.abs(convolve2D(channel_avg_h, prewitt_h))

# convert np.ndarray to uint8
abs_grad_h = abs_grad_h.astype(np.uint8)

# threshold
_, abs_grad_h = cv2.threshold(abs_grad_h, 30, 255,cv2.THRESH_BINARY)

cv2.imshow('Img_LP_h', abs_grad_h)
cv2.imwrite('Img_LP_h.jpg', abs_grad_h)

# high passed image
high_passed = img_resized - img_lpf      # subtract low pass from original 
cv2.imwrite('Img_HP.jpg', high_passed)
cv2.imshow('sss', high_passed)


k = cv2.waitKey(0)

if k == ord('e'):         # exit by pressing 'e' 
    cv2.destroyAllWindows()