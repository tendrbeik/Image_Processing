# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:08:28 2024

@author: abdra
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utility import util

# read the input image
img = cv2.imread("images/LR_3.jpg")
# convert from BGR to RGB so we can plot using matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# disable x & y axis
plt.axis('off')
# show the image
plt.imshow(img)
plt.show()
# get the image shape
rows, cols, dim = img.shape
# transformation matrix for Shearing
# shearing applied to x-axis
M = np.float32([[1, -0.5, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])
# shearing applied to y-axis
# M = np.float32([[1,   0, 0],
#             	  [0.5, 1, 0],
#             	  [0,   0, 1]])
# apply a perspective transformation to the image                
sheared_img = cv2.warpPerspective(img,M,(int(cols*0.62),int(rows*1)))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.imshow(sheared_img)
plt.show()
# save the resulting image to disk
plt.imsave("LR_3_anti_sheared.jpg", sheared_img)

#Теперь повысим контраст
gray_img = cv2.cvtColor(sheared_img, cv2.COLOR_BGR2GRAY)
kernel1 = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
filtered_image1 = cv2.filter2D(gray_img, -1, kernel1)
plt.imshow(gray_img, cmap='gray')
plt.show()

#Сделаем ВЧ-фильтрацию 

r = 1
rows, cols = gray_img.shape
crow, ccol = np.uint32((rows / 2, cols / 2))
# create a mask first, center square is 1, remaining all zeros
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - r:crow + r, ccol - r:ccol + r] = 0

# вычисляем фурье-образ
dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# применяем маску и делаем обратное преобразование Фурье
dft_shift_masked = dft_shift * mask
f_ishift = np.fft.ifftshift(dft_shift_masked)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[..., 0], img_back[..., 1])

magnitude_dft_shift = 20 * np.log(
    cv2.magnitude(dft_shift[..., 0], dft_shift[..., 1]))
magnitude_dft_shift_masked = 20 * np.log(
    cv2.magnitude(dft_shift_masked[..., 0], dft_shift_masked[..., 1]))

# вывод
plt.figure(figsize=(20, 20))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Исходное изображение'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_dft_shift, cmap='gray')
plt.title('Амплитудный спектр'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back, cmap='gray')
plt.title('Восстановленное изображение'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(magnitude_dft_shift_masked, cmap='gray')
plt.title('Обрезанный спектр'), plt.xticks([]), plt.yticks([])
plt.show()