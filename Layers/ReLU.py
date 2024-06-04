import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        output_tensor = np.maximum(0, input_tensor)

        return output_tensor

    def backward(self, error_tensor):
        # Compute the derivative of ReLU: 1 for input_tensor > 0, 0 otherwise
        relu_derivative = (self.input_tensor > 0).astype(np.float32)

        error_tensor_previous = error_tensor * relu_derivative

        return error_tensor_previous
