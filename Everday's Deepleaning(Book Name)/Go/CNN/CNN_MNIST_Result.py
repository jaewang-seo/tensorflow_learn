from cv2 import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
img_ori = cv2.imread('Go/CNN/images/test_2.jpg')

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
gray = np.where(gray >160, 255, gray)
gray = gray.astype('float64') / 255
plt.imshow(gray, cmap='gray')
plt.show()
gray = gray.reshape((1, 28, 28, 1))


model = load_model('CNN_MNIST_MODEL/24-0.0276.hdf5')

num = model.predict_classes(gray)
print(num[0])
print("OK")
