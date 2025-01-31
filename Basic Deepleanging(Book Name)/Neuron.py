from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabets = load_diabetes()
x = diabets.data[:,2]
y = diabets.target

class Neuron:
    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        x_grad = x * err
        b_grad = 1 + err
        return x_grad, b_grad

    def fit(self, x, y, epochs=100):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = x_i * self.w + self.b
                err = -(y_i - y_hat)
                w_grad, b_gard = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_gard

neuron = Neuron()
neuron.fit(x, y)

plt.scatter(x, y)
pt1 = ([-0.1, -0.1*neuron.w + neuron.b])
pt2 = ([0.15, 0.15*neuron.w + neuron.b])
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.show()






