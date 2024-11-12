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
image_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# #Выводим исходное изображение
# plt.figure(figsize=(15,20))
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.show()

# Проводим цветовую сегментацию, чтобы выделить бутылку на изображении
# Осуществление сегментации

#Алгоритм возрастания областей
# определяем координаты начальных точек
#Применяем гауссовское размытие
kernel55 = np.ones((5, 5), np.float32) / 25
kernel77 = np.ones((7, 7), np.float32) / 49
kernel88 = np.ones((8, 8), np.float32) / 64

# blur_image = cv.filter2D(image_hsv, -1, kernel55)
blur_image = image_hsv
seeds = [(800, 590), (400, 545), (1000, 550), (1100, 550),(1120, 590),(945, 495)]
# координаты для графика
x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
# порог похожести цвета региона
threshold = 94
# threshold = 100
# находим сегментацию используя метод из segmentation_utils
segmented_region = segmentation_utils.region_growingHSV(blur_image, seeds, threshold)
# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(img, img, mask=segmented_region)
# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(blur_image, cv.COLOR_HSV2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()
#Данный алгоритм позволил выделить бутылку на изображении
#Однако результат не очень хорош, а так же метод весьма громоздок
#в применении

# Алгоритм водораздела
# Бинаризируем изображение
binary_image = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
# Определяем карту расстояний
distance_map = ndimage.distance_transform_edt(binary_image)
# Определяем локальные максимумы
local_max = peak_local_max(distance_map, min_distance=20, labels=binary_image)

# 4 Каждому минимуму присваивается метка и начинается заполнение бассейнов метками
#Заменим следующую строку
#markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]

mask = np.zeros(distance_map.shape, dtype=bool)
mask[tuple(local_max.T)] = True
markers, _ = ndimage.label(mask)

labels = watershed(-distance_map, markers, mask=binary_image)
# построим результаты работы алгоритма
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(distance_map), cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(np.uint8(labels))
plt.show()

#Выведем количество выделенных областей
print(np.unique(labels))

# Найдем границы контуров и положим в маску все кроме метки 0
mask1 = np.zeros(image_gray.shape[0:2], dtype="uint8")
total_area = 0
for label in np.unique(labels):
    if label < 7 or label > 14 or (label <= 13 and label >=8):
        continue
    # Create a mask
    mask = np.zeros(image_gray.shape, dtype="uint8")
    mask[labels == label] = 255
    mask1 = mask1 + mask

    # Find contours and determine contour area
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv.contourArea)
    area = cv.contourArea(c)
    total_area += area
    cv.drawContours(image_gray, [c], -1, (36,255,12), 1)

result = cv.bitwise_and(img, img, mask=mask1)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mask1, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()
#Вывод: сегментация методом водораздела не позволила выделить только бутылку на изображении




#Метод деления показал, какие участки изображения имеют схожие цвета
qt = segmentation_utils.QTree(stdThreshold = 0.25, minPixelSize = 4,img = img.copy())
qt.subdivide()
tree_image = qt.render_img(thickness=1, color=(0,0,0))

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.show()



# Методы кластеризации. K-средних
# Преобразуем изображение в оттенках серого в одномерный массив
pixels = image_gray.reshape(-1, 1)
# Задаем число кластеров для сегментации
K = 10
# С помощью библиотеки sklearn.cluster import KMeans проводим кластеризацию по яркости
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_
print (np.uint8(cluster_centers))
# Каждому пикселю назначаем значение из центра кластера
segments = np.uint8(cluster_centers[labels].reshape(image_gray.shape))
mask = np.copy(segments)
# Удалим самые яркие пиксели
segments[segments!=10] = 0
result = cv.bitwise_and(img, img, mask=segments)
# Отобразим избражения
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='Set3')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()
#Данный метод позволил выделить бутылку на изображении, но не полностью

## Методы кластеризации. Сдвиг среднего (Mean shift)
# Сглаживаем чтобы уменьшить шум
blur_image = cv.medianBlur(img, 3)
# Выстраиваем пиксели в один ряд и переводим в формат с правающей точкой
flat_image = np.float32(blur_image.reshape((-1,3)))

# Используем meanshift из библиотеки sklearn
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=100)
ms = MeanShift(bandwidth=bandwidth, max_iter=100, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# получим количество сегментов
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# получим средний цвет сегмента
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# Для каждого пискеля проставим средний цвет его сегмента
mean_shift_image = avg[labeled].reshape((img.shape))
# Маской скроем один из сегментов
mask1 = mean_shift_image[:,:,0]
print(np.unique(mask1))
mask1[mask1!=44] = 0
mean_shift_with_mask_image = cv.bitwise_and(img, img, mask=mask1)
# Построим изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mean_shift_image, cmap='Set3')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(mean_shift_with_mask_image, cv.COLOR_BGR2RGB))
plt.show()
#Сдвиг среднего позволил выделить бутылку на изображении

# #Меняем кодировку
# result = cv.cvtColor(result, cv.COLOR_HSV2RGB)
# #Сохраняем результат
# plt.imsave("images/lr6_res.png", result)