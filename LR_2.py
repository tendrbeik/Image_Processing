# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 07:05:59 2024

@author: anatoly
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

def my_sum(arr,hist):
    shape = arr.shape
    result = np.zeros(shape,dtype=int)
    range1 = range(shape[0]-1)
    range2 = range(shape[1]-1)
    
    matrix = np.zeros(hist.shape)
    
    for i in range(255):
        for j in range(i):
            matrix[i] += hist[j]
    
    for i in range1:
        for j in range2:
                result[i][j] = matrix[arr[i][j]]
                
    return result

    
    
image1 = cv.imread('images/lenna_bad.png')
image2 = cv.imread('images/winter_cat.png')
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
rgb_image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# lut = lambda i: 255 * ((i - np.min(i)) / (np.max(i) - np.min(i)))
# result_image = lut(gray_image1)

# range = [0, 256]
# gs = plt.GridSpec(2, 2)
# plt.figure(figsize=(10, 8))
# plt.subplot(gs[0])
# plt.imshow(gray_image1, cmap='gray')
# plt.subplot(gs[1])
# plt.imshow(result_image, cmap='gray')
# plt.subplot(gs[2])
# plt.hist(gray_image1.reshape(-1), 256, irange)
# plt.subplot(gs[3])
# plt.hist(result_image.reshape(-1), 256, irange)
# plt.show()

channels = [0]
histSize = [256]
irange = [0, 256]

hist1 = cv.calcHist([gray_image1], channels, None, histSize, irange)
hist2 = cv.calcHist([gray_image2], channels, None, histSize, irange)

lut = lambda i: 255 * (my_sum(i,hist1)/sum(hist1))
result_image = lut(gray_image1)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.subplot(gs[1])
plt.imshow(result_image, cmap='gray')
plt.subplot(gs[2])
plt.hist(gray_image1.reshape(-1), 256, irange)
plt.subplot(gs[3])
plt.hist(result_image.reshape(-1), 256, irange)
plt.show()

lut = lambda i: 255 * (my_sum(i,hist2)/sum(hist2))
result_image = lut(gray_image2)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image2, cmap='gray')
plt.subplot(gs[1])
plt.imshow(result_image, cmap='gray')
plt.subplot(gs[2])
plt.hist(gray_image2.reshape(-1), 256, irange)
plt.subplot(gs[3])
plt.hist(result_image.reshape(-1), 256, irange)
plt.show()
