import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - (self.learning_rate * gradient_tensor)

        return updated_weight


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        self.velocity = self.momentum_rate * self.velocity + self.learning_rate * gradient_tensor
        updated_weight = weight_tensor - self.velocity

        return updated_weight


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.t = 0  # Time step t
        self.m = None  # First moment vector
        self.v = None  # Second moment vector

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)

        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        self.t += 1

        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (np.square(gradient_tensor))

        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)

        updated_weight = weight_tensor - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))

        return updated_weight
