import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_indices = None

    def forward(self, input_tensor):
        self.last_shape = input_tensor.shape
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        # The output tensor
        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1

        output_tensor = np.zeros((batch_size, channels, out_height, out_width))

        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        # Perform the pooling operation using numpy's max function
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width

                pool_region = input_tensor[:, :, h_start:h_end, w_start:w_end]

                max_val = np.max(pool_region, axis=(2, 3))

                output_tensor[:, :, i, j] = max_val

                # Find the indices of the max values in the pooling region
                max_indices = np.argmax(pool_region.reshape(batch_size, channels, -1), axis=2)
                x_indices, y_indices = np.unravel_index(max_indices, (pool_height, pool_width))

                # Store the max indices for the backward pass
                self.max_indices[:, :, i, j, 0] = x_indices
                self.max_indices[:, :, i, j, 1] = y_indices

        return output_tensor

    def backward(self, error_tensor):
        batch_size, channels, out_height, out_width = error_tensor.shape
        _, _, height, width = self.last_shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        # Initialize the gradient input tensor
        gradient_input = np.zeros(self.last_shape)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width

                # Get the max indices from the forward pass
                x_indices = self.max_indices[:, :, i, j, 0]
                y_indices = self.max_indices[:, :, i, j, 1]

                # Propagate the error tensor to the positions of the max values
                for batch in range(batch_size):
                    for channel in range(channels):
                        gradient_input[batch, channel, h_start:h_end, w_start:w_end][
                            x_indices[batch, channel], y_indices[batch, channel]] += error_tensor[batch, channel, i, j]

        return gradient_input