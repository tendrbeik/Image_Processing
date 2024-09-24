import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#Загружаем изображение
img = cv.imread("images/lr5.png")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Проводим цветовую сегментацию, чтобы выделить бутылку на изображении
