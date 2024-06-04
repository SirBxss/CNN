import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size, weights_initializer=None, bias_initializer=None):
        super().__init__()
        self.trainable = True
        # Initialize weights (including biases) uniformly in the range [0, 1)
        # Was considering separately --> simpler implementation
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        self.input_tensor = None
        self.grad_weights = None
        self._optimizer = None

        self.input_size = input_size
        self.output_size = output_size

        if weights_initializer is not None:
            self.weights = weights_initializer.initialize((input_size + 1, output_size), input_size, output_size)
        else:
            # Default initialization: Uniform random between [0, 1)
            self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        if bias_initializer is not None:
            self.weights[-1] = bias_initializer.initialize((output_size,), input_size, output_size)
        else:
            # Default initialization: Uniform random between [0, 1)
            self.weights[-1] = np.random.uniform(0, 1, (output_size,))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        input_tensor_with_bias = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))

        # Compute the output tensor using the dot product of the input tensor with the combined weights
        output_tensor = np.dot(input_tensor_with_bias, self.weights)

        return output_tensor

    def backward(self, error_tensor):
        input_tensor_with_bias = np.hstack((self.input_tensor, np.ones((self.input_tensor.shape[0], 1))))

        # Calculate gradients for the combined weights and bias -_-
        self.grad_weights = np.dot(input_tensor_with_bias.T, error_tensor)
        print(f"grad weight :   {self.grad_weights}")

        # Calculate error tensor for the previous layer --> Watch for the bias
        error_tensor_previous = np.dot(error_tensor, self.weights[:-1].T)
        print(f"pre_error :  {error_tensor_previous}")

        # If an optimizer is set, update the combined weights using the optimizer's calculate_update method
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.grad_weights)

        return error_tensor_previous

    def initialize(self, weights_initializer, bias_initializer):
        if weights_initializer is not None:
            self.weights[:-1] = weights_initializer.initialize((self.input_size, self.output_size), self.input_size,
                                                               self.output_size)
        else:
            # Default initialization: Uniform random between [0, 1)
            self.weights[:-1] = np.random.uniform(0, 1, (self.input_size, self.output_size))

        if bias_initializer is not None:
            self.weights[-1] = bias_initializer.initialize((self.output_size,), self.input_size, self.output_size)
        else:
            # Default initialization: Uniform random between [0, 1)
            self.weights[-1] = np.random.uniform(0, 1, (self.output_size,))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @property
    def gradient_weights(self):
        # Return the gradient for weights from the combined gradients
        return self.grad_weights

    # @property
    # def gradient_biases(self):
    #     # Return the gradient for biases from the combined gradients
    #     return self.grad_weights[-1:]
    #
    # @property
    # def weights_matrix(self):
    #     # Return the weights part of the combined weights
    #     return self.weights[:-1]
    #
    # @property
    # def biases(self):
    #     # Return the biases part of the combined weights
    #     return self.weights[-1].reshape(1, -1)
