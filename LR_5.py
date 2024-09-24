import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import colors

#Метод цветовой сегментации изображения
def segment_image(image):
    ''' Attempts to segment the whale out of the provided image '''

    # Convert the image into HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    #Цвета бутылки
    lower_bottle = (50,90,0)
    upper_bottle = (150,255,255)

    #Цвета апельсина на бутылке
    mask1 = cv.inRange(hsv_image, lower_bottle, upper_bottle)
    
    #
    lower_orange = (150,150,0)
    upper_orange = (200,250,250)
    
    mask2 = cv.inRange(hsv_image, lower_orange, upper_orange)
    
    final_mask = mask1 + mask2

    result = cv.bitwise_and(image, image, mask=final_mask)
    
    #Цвета синей части кружева под с бутылкой
    lower_bottle = (110,0,150)
    upper_bottle = (120,200,255)
    mask1 = cv.inRange(hsv_image, lower_bottle, upper_bottle)
    mask_inv = cv.bitwise_not(mask1)
    result = cv.bitwise_and(result, result, mask=mask_inv)

    #Clean up the segmentation using a blur
    blur = cv.GaussianBlur(result, (5, 5), 0)
    return blur

#Загружаем изображение
img = cv.imread('images/lr5.jpg')
image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
hsv_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

# Строим график для исходного изображения
h, s, v = cv.split(hsv_image)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

#Проводим цветовую сегментацию, чтобы выделить бутылку на изображении
#Осуществление сегментации
result = segment_image(image_rgb)

#Вывод результата
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

#Меняем кодировку
result = cv.cvtColor(result, cv.COLOR_HSV2RGB)
#Сохраняем результат
plt.imsave("images/lr5_res.png", result)

