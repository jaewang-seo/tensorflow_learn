# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# import numpy as np
# import tensorflow as tf

# df = pd.read_csv('dataset/pima-indians-diabetes.csv',
#                  names=['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'prdigree', 'age', 'class'])

# print(df.head(5))
# print(df.describe())
# print(df.info())
# print(df[['pregnant', 'class']])

# 데이터 가공하기
# print(df[['pregnant', 'class']].groupby(
#     ['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))


# plt.figure(figsize=(12, 12))
# sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5,
#             linecolor='white', annot=True, cmap=plt.cm.gist_heat)
# plt.show()

# grid = sns.FacetGrid(df, col='class')
# grid.map(plt.hist, 'plasma')
# plt.show()

# print(np.random.seed(seed))
# tf.random.set_seed(seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

dataset = np.loadtxt('dataset/pima-indians-diabetes.csv', delimiter=",")
X = dataset[:, :8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=1000, batch_size=10)

print(model.evaluate(X, Y))
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
