from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras import utils
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

np.random.seed(3)
tf.random.set_seed(3)

df = pd.read_csv('dataset/iris.csv', names=['sepal_length',
                                            'sepal_width', 'petal_length', 'petal_width', "species"])
# print(df.head(5))

# sns.pairplot(df, hue='species')
# plt.show()

dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]


e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

Y_encoded = utils.to_categorical(Y)


model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X, Y_encoded, epochs=100, batch_size=1)
# print("OK")
