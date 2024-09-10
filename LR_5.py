# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:06:09 2024

@author: anatoly
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

#Метод цветовой сегментации изображения
def segment_image(image):
    ''' Attempts to segment the whale out of the provided image '''

    # Convert the image into HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    #Set the gray range
    upper_cat = (50,200,200)
    lower_cat = (1,0,0)

    # Apply the gray mask
    mask1 = cv.inRange(hsv_image, lower_cat, upper_cat)
    
    lower_white = (100,0,0)
    upper_white = (110,255,255)
    
    mask2 = cv.inRange(hsv_image, lower_white, upper_white)
    
    final_mask = mask1 + mask2

    result = cv.bitwise_and(image, image, mask=final_mask)

    #Clean up the segmentation using a blur
    blur = cv.GaussianBlur(result, (7, 7), 0)
    return blur

#Загрузка исходного изображения
img = cv.imread('images/winter_cat.png')
image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
hsv_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

# h, s, v = cv.split(hsv_image)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
# pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()
# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()

#Осуществление сегментации
result = segment_image(image_rgb)

#Вывод результата
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
