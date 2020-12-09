from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()

# plt.scatter(diabetes.data[:,2], diabetes.target)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

x = diabetes.data[:,2]
y = diabetes.target
w = 1.0
w_inc = w+0.1
b = 1.0
# y_hat = w * x[0] + b
# y_hat2 = w_inc * x[0] + b
# w_rate = (y_hat2-y_hat) / (w_inc - w)

for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w_rate * err + w
    b = 1 * err + b

plt.scatter(x, y)
pt1 = ()
print("OK")