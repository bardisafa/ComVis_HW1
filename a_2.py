
import cv2
import numpy as np
import argparse

alpha_slider_max = 360
title_window = 'Rotation'


# Define rotation function:
def rotation(img, angle):
    h, w = img.shape[:2]  # height and width of image
    img_center = (
    w / 2, h / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(img_center, angle, 1.)

    # calculate the absolute of cos and sin
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds and get ceiling for odd numbers
    h_new = int(np.round(h * abs_cos + w * abs_sin))
    h_new = int(np.ceil(h_new / 2) * 2)

    w_new = int(np.round(h * abs_sin + w * abs_cos))
    w_new = int(np.ceil(w_new / 2) * 2)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += w_new / 2 - img_center[0]
    rotation_mat[1, 2] += h_new / 2 - img_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(img, rotation_mat, (w_new, h_new))
    return rotated_mat, rotation_mat


def on_trackbar(val):
    point = [200, 200]
    v = [point[0], point[1], 1]

    aa, M = rotation(image, val)

    rotated_point = np.round(np.dot(M, v))  # point coordinates after rotation
    # print(rotated_point)

    newX = aa.shape[1]
    newY = aa.shape[0]
    oldX = image.shape[1]
    oldY = image.shape[0]

    # Resizing ratios:
    Rx = newX / oldX
    Ry = newY / oldY

    imgg = cv2.resize(image, (int(newX), int(newY)))  # Resized original image

    resized_point = [point[0] * Rx, point[1] * Ry]  # Resized point
    resized_point = np.round(resized_point)
    # print(resized_point)

    # Concatinate original image and rotated image:
    cc = np.concatenate((imgg, aa), axis=1)

    # Draw line
    cv2.line(cc, (int(resized_point[0]), int(resized_point[1])),
             (int(rotated_point[0]) + imgg.shape[1], int(rotated_point[1])), (0, 0, 255), 2)
    r = 1280.0 / cc.shape[1]
    dim = (1280, int(cc.shape[0] * r))

    final_img = cv2.resize(cc, dim)

    cv2.imshow(title_window, final_img)




parser = argparse.ArgumentParser(description='Code for Adding a Trackbar.')
parser.add_argument('--input1', help='Path to the first input image.', default='space.jpg')

args = parser.parse_args()

image = cv2.imread(cv2.samples.findFile(args.input1))



cv2.namedWindow(title_window)
trackbar_name = 'Angle'
cv2.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)

# Show some stuff
on_trackbar(0)
# Wait until user press some key
cv2.waitKey()
