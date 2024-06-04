from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        input_tensor = input_tensor.reshape(self.input_shape[0], -1)

        return input_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.input_shape)
        return error_tensor
