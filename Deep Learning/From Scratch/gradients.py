import numpy as np


class VanillaGradient:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        return params - self.lr * grads


class MomentumGradient:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0

    def update(self, params, grads):
        self.v = self.beta * self.v + (1 - self.beta) * grads
        return params - self.lr * self.v


class NesterovGradient:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0

    def update(self, params, grads):
        lookahead = params - self.beta * self.v
        self.v = self.beta * self.v + (1 - self.beta) * grads
        return lookahead - self.lr * self.v


class AdaGrad:
    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.G = 0  # Sum of squared gradients

    def update(self, params, grads):
        self.G += grads ** 2
        return params - (self.lr / (np.sqrt(self.G) + self.epsilon)) * grads


class RMSProp:
    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.G = 0

    def update(self, params, grads):
        self.G = self.beta * self.G + (1 - self.beta) * grads ** 2
        return params - (self.lr / (np.sqrt(self.G) + self.epsilon)) * grads


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0  # First moment (momentum)
        self.v = 0  # Second moment (RMSProp)
        self.t = 0  # Time step

    def update(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)  # Bias correction
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat