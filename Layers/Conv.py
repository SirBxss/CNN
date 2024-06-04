from Layers.Base import BaseLayer
import numpy as np
from scipy.signal import *


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernel):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernel

        # Initialize weights and biases
        if len(convolution_shape) == 2:  # 1D convolution
            c, m = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernel, c, m))
        elif len(convolution_shape) == 3:  # 2D convolution
            c, m, n = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernel, c, m, n))
        else:
            raise ValueError("Invalid convolution shape")
        self.bias = np.random.uniform(0, 1, num_kernel)

        # self.weights = np.random.uniform(0, 1, (num_kernel, *convolution_shape))
        # self.bias = np.random.uniform(0, 1, num_kernel)

        self._gradient_weights = None
        self._gradient_bias = None
        self._bias_optimizer = None
        self.optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        b = input_tensor.shape[0]
        if len(self.convolution_shape) == 3:
            output_shape = (
                b,
                self.num_kernels,
                int(np.ceil(input_tensor.shape[2] / self.stride_shape[0])),
                int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))
            )
        elif len(self.convolution_shape) == 2:
            output_shape = (
                b,
                self.num_kernels,
                int(np.ceil(input_tensor.shape[2] / self.stride_shape[0]))
            )

        output_tensor = np.zeros(output_shape)

        for i in range(b):
            for j in range(self.num_kernels):
                conv_planes = np.sum([
                    correlate(
                        input_tensor[i, k],
                        self.weights[j, k],
                        mode='same',
                        method='direct'
                    )
                    for k in range(self.weights.shape[1])
                ], axis=0)

                if len(self.convolution_shape) == 3:
                    conv_planes = conv_planes[::self.stride_shape[0], ::self.stride_shape[1]]
                elif len(self.convolution_shape) == 2:
                    conv_planes = conv_planes[::self.stride_shape[0]]

                output_tensor[i, j] = conv_planes + self.bias[j]

        return output_tensor

        # if len(input_tensor.shape) == 3:
        # 1D Convolution
        #     b, c, y = input_tensor.shape
        #     stride_y = self.stride_shape[0]
        #     pad_y = (self.convolution_shape[1] - 1) // 2
        #     padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_y, pad_y)), mode='constant')
        #     output_y = (y + 2 * pad_y - self.convolution_shape[1]) // stride_y + 1
        #     output = np.zeros((b, self.num_kernels, output_y))
        #     for i in range(b):
        #         for j in range(self.num_kernels):
        #             for k in range(c):
        #                 output[i, j] += correlate(padded_input[i, k], self.weights[j, k], mode='valid')[::stride_y]
        #             output[i, j] += self.bias[j]
        # elif len(input_tensor.shape) == 4:
        #     # 2D Convolution
        #     b, c, y, x = input_tensor.shape
        #     stride_y, stride_x = self.stride_shape
        #     print(f"stride x: {stride_x} stride y: {stride_y}")
        #     print(f"convolution shape : {self.convolution_shape}")
        #     pad_y = (self.convolution_shape[1] - 1) // 2
        #     print(f"pad y: {pad_y}")
        #     pad_x = (self.convolution_shape[2] - 1) // 2
        #     print(f"pad x: {pad_x}")
        #     padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant')
        #     output_y = (y + 2 * pad_y - self.convolution_shape[1]) // stride_y + 1
        #     output_x = (x + 2 * pad_x - self.convolution_shape[2]) // stride_x + 1
        #     print(f"output x: {output_x}")
        #     output = np.zeros((b, self.num_kernels, output_y, output_x))
        #     for i in range(b):
        #         for j in range(self.num_kernels):
        #             for k in range(c):
        #                 output[i, j] += correlate(padded_input[i, k], self.weights[j, k], mode='valid')[::stride_y, ::stride_x]
        #             output[i, j] += self.bias[j]
        #     return output

    def backward(self, error_tensor):
        gradient_input = np.zeros_like(self.input_tensor)
        new_weights = np.copy(self.weights)

        # Ensure stride values are integers and handle both 1D and 2D cases
        stride_shape_0 = int(self.stride_shape[0])
        stride_shape_1 = int(self.stride_shape[1]) if len(self.stride_shape) > 1 else 1  # Default to 1 for 1D case

        if len(self.convolution_shape) == 3:  # 2D Convolution
            temp_gradient_weights = np.zeros(
                (error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1], self.weights.shape[2],
                 self.weights.shape[3])
            )

            padded_input = np.pad(
                self.input_tensor,
                ((0, 0), (0, 0),
                 (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                 (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)),
                mode='constant'
            )

            if self.convolution_shape[2] % 2 == 0:
                padded_input = padded_input[:, :, :, :-1]
            if self.convolution_shape[1] % 2 == 0:
                padded_input = padded_input[:, :, :-1, :]

            for batch in range(error_tensor.shape[0]):
                for out_ch in range(error_tensor.shape[1]):
                    temp = resample(error_tensor[batch, out_ch], error_tensor[batch, out_ch].shape[0] * stride_shape_0,
                                    axis=0)
                    temp = resample(temp, error_tensor[batch, out_ch].shape[1] * stride_shape_1, axis=1)
                    temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

                    if stride_shape_1 > 1:
                        temp[:, ::stride_shape_1] = 0
                    if stride_shape_0 > 1:
                        temp[::stride_shape_0, :] = 0

                    for in_ch in range(self.input_tensor.shape[1]):
                        temp_gradient_weights[batch, out_ch, in_ch] = correlate(padded_input[batch, in_ch], temp,
                                                                                mode='valid')

            self._gradient_weights = temp_gradient_weights.sum(axis=0)

        elif len(self.convolution_shape) == 2:  # 1D Convolution
            temp_gradient_weights = np.zeros(
                (error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1], self.weights.shape[2])
            )

            padded_input = np.pad(
                self.input_tensor,
                ((0, 0), (0, 0),
                 (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2)),
                mode='constant'
            )

            if self.convolution_shape[1] % 2 == 0:
                padded_input = padded_input[:, :, :-1]

            for batch in range(error_tensor.shape[0]):
                for out_ch in range(error_tensor.shape[1]):
                    temp = resample(error_tensor[batch, out_ch], error_tensor[batch, out_ch].shape[0] * stride_shape_0,
                                    axis=0)
                    temp = temp[:self.input_tensor.shape[2]]

                    if stride_shape_0 > 1:
                        temp[::stride_shape_0] = 0

                    for in_ch in range(self.input_tensor.shape[1]):
                        temp_gradient_weights[batch, out_ch, in_ch] = correlate(padded_input[batch, in_ch], temp,
                                                                                mode='valid')

            self._gradient_weights = temp_gradient_weights.sum(axis=0)

        new_weights = np.transpose(new_weights, (1, 0, 2, 3)) if len(self.convolution_shape) == 3 else np.transpose(
            new_weights, (1, 0, 2))

        for batch in range(error_tensor.shape[0]):
            for out_ch in range(new_weights.shape[0]):
                ch_conv_out = []

                for in_ch in range(new_weights.shape[1]):
                    temp = resample(error_tensor[batch, in_ch], error_tensor[batch, in_ch].shape[0] * stride_shape_0,
                                    axis=0)
                    if len(self.convolution_shape) == 3:
                        temp = resample(temp, error_tensor[batch, in_ch].shape[1] * stride_shape_1, axis=1)
                        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                    elif len(self.convolution_shape) == 2:
                        temp = temp[:self.input_tensor.shape[2]]

                    if stride_shape_1 > 1 and len(self.convolution_shape) == 3:
                        temp[:, ::stride_shape_1] = 0
                    if stride_shape_0 > 1:
                        temp[::stride_shape_0] = 0

                    ch_conv_out.append(convolve(temp, new_weights[out_ch, in_ch], mode='same', method='direct'))

                gradient_input[batch, out_ch] = np.sum(ch_conv_out, axis=0)

        if len(self.convolution_shape) == 3:
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        elif len(self.convolution_shape) == 2:
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, opt):
        self._gradient_weights = opt

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, opt):
        self._gradient_bias = opt

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, opt):
        self._bias_optimizer = opt
