import cv2  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Загружаем изображение
# rgb_img = cv2.imread('images/Pear.png') 
rgb_img = cv2.imread('images/findWolley.jpg') 
# Преобразуем изображение в оттенки серого 
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
# Показываем результат на изображении
#plt.figure(figsize=(8,8))
#plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))

# Загружаем шаблон  
# template = cv2.imread('images/PearTmpl.png')
template = cv2.imread('images/wolley2.jpg')
#template = cv2.imread('images/man2.jpg')

#Уже не нужно
# # Преобразуем в оттенки серого
# gray_templ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# # Показываем результат
# #plt.figure()
# #plt.imshow(cv2.cvtColor(gray_templ, cv2.COLOR_GRAY2RGB))

#Создаём шаблон в оттенках серого
# Преобразуем и вносим небольшие изменения в шаблон
scale = 1 # масштаб изменения размеров
scBr = 1 # коэффициент изменения яркости

template_scale = cv2.resize(np.uint8(scBr*cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)),
           (int(template.shape[1]*scale), int(template.shape[0]*scale)), interpolation = cv2.INTER_AREA)

# #Пока уберём SIFT и попробуем другие алгоритмы для выделения особых точек на изображении
# # Создаем детектор особых точек
# sift = cv2.SIFT_create()
# # sift = cv2.xfeatures2d.SIFT_create() # В зависимости от версии opencv может работать эта команда

# # Запускаем детектор на изображении и на шаблоне
# # Метод возвращает список особых точек и их дескрипторов
# k_1, des_1 = sift.detectAndCompute(gray_img, None)
# k_2, des_2 = sift.detectAndCompute(template_scale, None)

# img2 = cv2.drawKeypoints(gray_img, k_1, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()
# img2 = cv2.drawKeypoints(template_scale, k_2, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()


# Initiate ORB detector
#orb = cv2.ORB_create()
orb = cv2.ORB_create(nfeatures=1000000, edgeThreshold=0)

# find the keypoints with ORB
kp1 = orb.detect(gray_img,None)
# compute the descriptors with ORB
k_1, des_1 = orb.compute(gray_img, kp1)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(gray_img, kp1, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

#Теперь найдём точки для шаблона и отобразим их
kp2 = orb.detect(template_scale,None)
k_2, des_2 = orb.compute(template_scale, kp2)

img2 = cv2.drawKeypoints(template_scale, kp2, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

#Теперь обнаружим на изображении все груши
bf = cv2.BFMatcher(cv2.NORM_L1)
matches = bf.knnMatch(des_1, des_2, k=2)

# Лучшие пары особых точек отбираются с использованием теста отношения правдоподобия
good = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])

# построим совпадения на изображении
image_with_knn_matches = cv2.drawMatchesKnn(gray_img,k_1,template_scale,k_2,good,None,flags=2)
plt.figure(figsize=(15,15))
plt.imshow(cv2.cvtColor(image_with_knn_matches, cv2.COLOR_BGR2RGB))

#
points = np.array([(0, 0)])
for i in good:
    points = np.append(points, [k_1[i[0].queryIdx].pt], axis=0)

points = points[1:len(points)]

# Определяем ширину окна и запускаем алгоритм кластеризации
bandwidth = estimate_bandwidth(points, quantile=0.4)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False)
cl = ms.fit_predict(points)

# Формируем кластеры особых точек
labels_unique = np.unique(ms.labels_)
kp = []
for i in labels_unique:
    kp.append(points[cl==i])

# Определяем центры кластеров, но только если в кластере содержится более 10 точек
cen = []
for i in kp:
    if len(i)>=3:
        cen.append(np.mean(i, axis=0).astype(np.uint16))

# Вокруг выделенных центров обводим прямоугольники с размерами шаблона
plot_img = rgb_img.copy()
h, w = (template.shape[0],template.shape[1])

for pt in cen:
    cv2.rectangle(plot_img, (pt[0] - w, pt[1] - h),(pt[0] + w, pt[1] + h),(0,255,255), 8)  

# Отображаем результат на графике
plt.figure(figsize=(20,20))
plt.imshow(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))