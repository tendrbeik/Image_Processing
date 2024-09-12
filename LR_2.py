import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

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

    
#Загружаем исходные изображения и перводим их из цветных в серые
image1 = cv.imread('images/lenna_bad.png')
image2 = cv.imread('images/winter_cat.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

#Считаем гистограммы изображений
channels = [0]
histSize = [256]
irange = [0, 256]

hist1 = cv.calcHist([gray_image1], channels, None, histSize, irange)
hist2 = cv.calcHist([gray_image2], channels, None, histSize, irange)

#Осуществляем эквализацию изображения lenna_bad.png с помощью метода my_sum()
lut = lambda i: 255 * (my_sum(i,hist1)/sum(hist1))
result_image = lut(gray_image1)

#Выводим результаты эквализации изображения lenna_bad.png
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

#Осуществляем эквализацию изображения winter_cat.png
lut = lambda i: 255 * (my_sum(i,hist2)/sum(hist2))
result_image = lut(gray_image2)

#Выводим результаты эквализации изображения winter_cat.png
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