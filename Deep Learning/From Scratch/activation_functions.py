import numpy as np


class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        sig_x = self.forward(x)
        return sig_x * (1 - sig_x)


class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - self.forward(x) ** 2


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)


class LeakyReLU:
    def forward(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def backward(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)


class Swish:
    def forward(self, x):
        sig_x = Sigmoid().forward(x)
        return x * sig_x

    def backward(self, x):
        sig_x = Sigmoid().forward(x)
        swish_x = self.forward(x)
        return swish_x + sig_x * (1 - swish_x)


class ELU:
    def forward(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def backward(self, x, alpha=0.01):
        elu_x = self.forward(x, alpha)
        return np.where(x > 0, 1, elu_x + alpha)


class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x))  # Prevents overflow
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, x):
        softmax_x = self.forward(x)
        return softmax_x * (1 - softmax_x)