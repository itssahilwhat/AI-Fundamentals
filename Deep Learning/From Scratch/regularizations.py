import numpy as np


class L1Regularization:
    """ L1 Regularization (Lasso) """
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def penalty(self, weights):
        return self.lambda_ * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.lambda_ * np.sign(weights)


class L2Regularization:
    """ L2 Regularization (Ridge) """
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def penalty(self, weights):
        return 0.5 * self.lambda_ * np.sum(weights ** 2)

    def gradient(self, weights):
        return self.lambda_ * weights


class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            self.mask = (np.random.rand(*inputs.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            return inputs * self.mask
        return inputs

    def backward(self, grad_output):
        return grad_output * self.mask if self.mask is not None else grad_output


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience