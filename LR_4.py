# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:21:40 2024

@author: anatoly
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

def binarization(image, threshold):
    shape = image.shape
    result = np.zeros(shape,dtype=int)
    for i in range(shape[0]-1):
        for j in range(shape[1]-1):
            if image[i][j] > threshold:
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result

#Загрузим исходное изображение
img = cv.imread('images/winter_cat.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Выведем исходное изображение с гистограммой
channels = [0]
histSize = [256]
irange = [0, 256]

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(15, 15))
plt.subplot(gs[0])
plt.imshow(gray_img, cmap='gray')
plt.subplot(gs[1])
plt.hist(gray_img.reshape(-1), 256, irange)
plt.show()

#Осуществим бинаризацию
lut = lambda i: binarization(i, 200)
result_image = lut(gray_img)

#Выведем результат
plt.figure(figsize=(10, 10))
plt.imshow(result_image, cmap='gray')
plt.title('Результат бинаризации'), plt.xticks([]), plt.yticks([])
plt.show()