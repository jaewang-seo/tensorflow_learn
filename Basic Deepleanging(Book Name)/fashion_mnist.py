import tensorflow as tf
import matplotlib.pyplot as plt
# print(tf.__version__)

(x_train_all, y_train_all), (x_test, y_test)= tf.keras.datasets.fashion_mnist.load_data()

plt.imshow(x_train_all[0], cmap='gray')
plt.show()
print(y_train_all[0])

from sklearn.model_selection import train_test_split
import numpy as np
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=43)
x_train = x_train.reshape(-1, 784)
print("OK")