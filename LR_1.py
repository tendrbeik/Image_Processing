import numpy as np

#Создаём массивы А и В
A = np.ones(3)*1
B = np.ones(3)*2
#Вычисляем значение выражения, выданного в качестве задания
print(np.add(A,B)*np.divide(-A,2))
