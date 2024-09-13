import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#Первый успешный результат
#Загружаем изображение
img = cv.imread("images/as.png")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Проводим бинаризацию
threshold = 194

ret1, thresh1 = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)
ret2, thresh2 = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

plt.figure(figsize=(12, 12))
plt.imshow(thresh1, 'gray', vmin=0, vmax=255)
plt.title("Binarization (threshold = %d)" % ret1)
plt.xticks([])
plt.yticks([])
plt.show()

#Применяем метод Кани
threshold1 = 10
threshold2 = 20
edges = cv.Canny(thresh1, threshold1, threshold2)

#Применяем инверсию
lut = lambda i: 255 - i
edges = lut(edges)
plt.figure(figsize=(20, 20))
plt.imshow(edges, 'gray')
plt.show()

#Применяем гауссовское размытие
kernel55 = np.ones((5, 5), np.float32) / 25
kernel77 = np.ones((7, 7), np.float32) / 49

filtered_image = cv.filter2D(edges, -1, kernel77)
# filtered_image = cv.medianBlur(thresh1, 5)
plt.figure(figsize=(20, 20))
plt.imshow(filtered_image, 'gray')
plt.show()

#Меняем кодировку
filtered_image = cv.cvtColor(filtered_image, cv.COLOR_GRAY2RGB)
#Сохраняем результат
plt.imsave("images/as2.png", filtered_image)

