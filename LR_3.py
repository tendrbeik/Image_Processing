# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:05:40 2024

@author: anatoly
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

#Загрузим исходное изображение
img = cv.imread('images/winter_cat.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Проведём зашумление изображения
#Шум Соль и Перец
noisy_image1 = util.add_salt_and_peper_noise(gray_img, 0.2)
#Гауссовский шум
noisy_image2 = util.add_gauss_noise(gray_img, 0, 0.2)

#Выведем исходное изображение и зашумлённые изображения
gs = plt.GridSpec(1, 3)
plt.figure(figsize=(30, 30))

plt.subplot(gs[0])
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.imshow(gray_img, cmap='gray')

plt.subplot(gs[1])
plt.title('Соль и Перец')
plt.xticks([]), plt.yticks([])
plt.imshow(noisy_image1, cmap='gray')

plt.subplot(gs[2])
plt.title('Гауссовский шум')
plt.xticks([]), plt.yticks([])
plt.imshow(noisy_image2, cmap='gray')
plt.show()

#Проведём НЧ фильтрацию и выведем результаты
r = 50
rows, cols = gray_img.shape
crow, ccol = np.uint32((rows / 2, cols / 2))
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - r:crow + r, ccol - r:ccol + r] = 1

# вычисляем фурье-образ
dft = cv.dft(np.float32(gray_img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# применяем маску и делаем обратное преобразование Фурье
dft_shift_masked = dft_shift * mask
f_ishift = np.fft.ifftshift(dft_shift_masked)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[..., 0], img_back[..., 1])

magnitude_dft_shift = 20 * np.log(
    cv.magnitude(dft_shift[..., 0], dft_shift[..., 1]))
magnitude_dft_shift_masked = 20 * np.log(
    cv.magnitude(dft_shift_masked[..., 0], dft_shift_masked[..., 1]))

# вывод
plt.figure(figsize=(20, 20))
plt.subplot(221), plt.imshow(gray_img, cmap='gray')
plt.title('Исходное изображение'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_dft_shift, cmap='gray')
plt.title('Амплитудный спектр изображения'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back, cmap='gray')
plt.title('Восстановленное изображение'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(magnitude_dft_shift_masked, cmap='gray')
plt.title('Обрезанный спектр'), plt.xticks([]), plt.yticks([])
plt.show()

#Загружаем исходное изображение и выводим его
img = cv.imread('images/LR_3.jpg')
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.imshow(img, cmap='gray')
plt.show()

#Попробуем повысить резкость изображения, но сначала надо бы сдвиг обратить как-нибудь


