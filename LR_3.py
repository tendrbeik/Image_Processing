import numpy as np
import cv2
import matplotlib.pyplot as plt

#Загружаем исходное изображение и преобразуем его в RGB формат
img = cv2.imread("images/LR_3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Показываем исходное изображение
plt.figure(figsize=(15, 15))
plt.imshow(img)
plt.show()

#Убираем эффект сдвига изображения по оси Х и выводим результат
rows, cols, dim = img.shape
M = np.float32([[1, -0.5, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])
          
sheared_img = cv2.warpPerspective(img,M,(int(cols*0.62),int(rows*1)))
plt.figure(figsize=(15, 15))
plt.imshow(sheared_img)
plt.show()

#Можно сохранить преобразованное изображение по желанию
#plt.imsave("LR_3_anti_sheared.jpg", sheared_img)

#Повышаем контраст изображения, чтобы оно стало чётче, и выводим результат
gray_img = cv2.cvtColor(sheared_img, cv2.COLOR_BGR2GRAY)
kernel1 = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
filtered_image1 = cv2.filter2D(gray_img, -1, kernel1)
plt.figure(figsize=(15, 15))
plt.imshow(filtered_image1, cmap='gray')
plt.show()
#Результат: разобрать, что на изображении не получается.

#Попробуем осуществить ВЧ-фильтрацию изображения, чтобы увидеть очертания объектов на изображении
r = 5
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

# выводим результаты фильтрации
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
#Результат: по очертаниям изображения не понятно, что на нём содержится.