import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
from utility import segmentation_utils

#Загружаем изображение
img = cv.imread('images/lr5.jpg')
image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# hsv_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

#Проводим цветовую сегментацию, чтобы выделить бутылку на изображении
#Осуществление сегментации
## Методы кластеризации. K-средних
# Преобразуем изображение в оттенках серого в одномерный массив
pixels = image_gray.reshape(-1, 1)
# Задаем число кластеров для сегментации
K = 3
# С помощью библиотеки sklearn.cluster import KMeans проводим кластеризацию по яркости
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_
print (np.uint8(cluster_centers))
# Каждому пикселю назначаем значение из центра кластера
segments = np.uint8(cluster_centers[labels].reshape(image_gray.shape))
# Удалим самые яркие пиксели
segments[segments==167] = 0
result = cv.bitwise_and(image_gray, image_gray, mask=segments)
# Отобразим избражения
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image_gray, cv.COLOR_GRAY2RGB))
plt.subplot(1, 3, 2)
plt.imshow(segments, cmap='Set3')
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()

# #Меняем кодировку
# result = cv.cvtColor(result, cv.COLOR_HSV2RGB)
# #Сохраняем результат
# plt.imsave("images/lr5_res.png", result)

#Метод деления показал, какие участки изображения имеют схожие цвета
# qt = segmentation_utils.QTree(stdThreshold = 0.25, minPixelSize = 4,img = img.copy())
# qt.subdivide()
# tree_image = qt.render_img(thickness=1, color=(0,0,0))

# plt.figure(figsize=(15,20))
# plt.subplot(1, 2, 1)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
# plt.show()