import numpy as np


class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        range_uniform_distribution = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, range_uniform_distribution, size=weights_shape)


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        range_uniform_distribution = np.sqrt(2 / fan_in)
        return np.random.normal(0, range_uniform_distribution, size=weights_shape)
