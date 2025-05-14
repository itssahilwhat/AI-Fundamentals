import numpy as np
from activation_functions import Sigmoid, Tanh, ReLU, LeakyReLU, Swish, ELU, Softmax
from gradients import VanillaGradient, MomentumGradient, NesterovGradient, AdaGrad, RMSProp, Adam
from regularizations import L1Regularization, L2Regularization, Dropout, EarlyStopping


class DeepNeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, optimizer, reg=None, dropout_rate=0.0):
        self.layer_sizes = layer_sizes
        self.activation_functions = [self.get_activation_func(af) for af in activation_functions]
        self.optimizer = optimizer
        self.regularization = reg
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None

        # Initialize Weights and Biases
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
        self.gradients = [np.zeros_like(w) for w in self.weights]
        self.velocities = [np.zeros_like(w) for w in self.weights]

    def get_activation_func(self, name):
        activation_map = {
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
            "relu": ReLU(),
            "leaky_relu": LeakyReLU(),
            "swish": Swish(),
            "elu": ELU(),
            "softmax": Softmax()
        }
        return activation_map[name]

    def forward(self, X):
        self.a = [X]
        self.z = []

        for i, (w, b, activation) in enumerate(zip(self.weights, self.biases, self.activation_functions)):
            z = np.dot(self.a[-1], w) + b
            self.z.append(z)
            a = activation.forward(z)

            if self.dropout and i < len(self.weights) - 1:
                a = self.dropout.forward(a, training=True)

            self.a.append(a)

        return self.a[-1]

    def backward(self, y_true):
        m = y_true.shape[0]
        deltas = [self.a[-1] - y_true]

        for i in reversed(range(len(self.weights) - 1)):
            dz = np.dot(deltas[0], self.weights[i + 1].T) * self.activation_functions[i].backward(self.z[i])
            if self.dropout:
                dz *= self.dropout.backward(np.ones_like(dz))
            deltas.insert(0, dz)

        for i in range(len(self.weights)):
            dw = np.dot(self.a[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            if self.regularization:
                dw += self.regularization.gradient(self.weights[i]) / m

            self.gradients[i] = dw
            self.biases[i] -= self.optimizer.update(self.biases[i], db)
            self.weights[i] -= self.optimizer.update(self.weights[i], dw)

    def compute_loss(self, y_true):
        loss = -np.sum(y_true * np.log(self.a[-1] + 1e-9)) / y_true.shape[0]

        if self.regularization:
            loss += self.regularization.penalty(self.weights)

        return loss

    def fit(self, X, y, epochs=100, batch_size=32, early_stopping=None):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]

            for i in range(0, len(X), batch_size):
                X_batch, y_batch = X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]
                self.forward(X_batch)
                self.backward(y_batch)

            loss = self.compute_loss(y)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

            if early_stopping and early_stopping.should_stop(loss):
                print("Early stopping triggered!")
                break

    def predict(self, X):
        return self.forward(X)