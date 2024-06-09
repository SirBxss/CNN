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
        '''Updates the parameters using the optimizer and returns the error tensor for the next layer'''

        gradient_input = np.zeros_like(self.input_tensor)
        new_weights = np.copy(self.weights)

        if len(self.convolution_shape) == 3:
            self._calculate_gradient_weights_3d(error_tensor)
            new_weights = self._rearrange_weights(new_weights, dims=4)
            gradient_input = self._calculate_gradient_input_3d(error_tensor, new_weights, gradient_input)
            self._calculate_gradient_bias_3d(error_tensor)
        elif len(self.convolution_shape) == 2:
            self._calculate_gradient_weights_1d(error_tensor)
            new_weights = self._rearrange_weights(new_weights, dims=3)
            gradient_input = self._calculate_gradient_input_1d(error_tensor, new_weights, gradient_input)
            self._calculate_gradient_bias_1d(error_tensor)

        self._update_weights_and_biases()

        return gradient_input

    def _calculate_gradient_weights_3d(self, error_tensor):
        temp_gradient_weights = np.zeros((error_tensor.shape[0], *self.weights.shape))

        padded_input = self._pad_input_3d()

        for batch in range(error_tensor.shape[0]):
            for out_ch in range(error_tensor.shape[1]):
                upsampled_error = self._upsample_error_3d(error_tensor[batch, out_ch])

                for in_ch in range(self.input_tensor.shape[1]):
                    temp_gradient_weights[batch, out_ch, in_ch] = correlate(
                        padded_input[batch, in_ch], upsampled_error, mode='valid')

        self.gradient_weights = temp_gradient_weights.sum(axis=0)

    def _pad_input_3d(self):
        conv_plane_out = []
        for batch in range(self.input_tensor.shape[0]):
            ch_conv_out = []
            for out_ch in range(self.input_tensor.shape[1]):
                padded_channel = np.pad(
                    self.input_tensor[batch, out_ch],
                    ((self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                     (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)),
                    mode='constant')

                if self.convolution_shape[2] % 2 == 0:
                    padded_channel = padded_channel[:, :-1]
                if self.convolution_shape[1] % 2 == 0:
                    padded_channel = padded_channel[:-1, :]

                ch_conv_out.append(padded_channel)

            conv_plane = np.stack(ch_conv_out, axis=0)
            conv_plane_out.append(conv_plane)

        return np.stack(conv_plane_out, axis=0)

    def _upsample_error_3d(self, error):
        temp = resample(error, error.shape[0] * self.stride_shape[0], axis=0)
        temp = resample(temp, error.shape[1] * self.stride_shape[1], axis=1)
        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

        if self.stride_shape[1] > 1:
            temp[:, 1::self.stride_shape[1]] = 0
        if self.stride_shape[0] > 1:
            temp[1::self.stride_shape[0], :] = 0

        return temp

    def _calculate_gradient_input_3d(self, error_tensor, new_weights, gradient_input):
        for batch in range(error_tensor.shape[0]):
            for out_ch in range(new_weights.shape[0]):
                ch_conv_out = []
                for in_ch in range(new_weights.shape[1]):
                    upsampled_error = self._upsample_error_3d(error_tensor[batch, in_ch])
                    convolved = convolve(
                        upsampled_error, new_weights[out_ch, in_ch], mode='same', method='direct')
                    ch_conv_out.append(convolved)

                gradient_input[batch, out_ch] = np.sum(ch_conv_out, axis=0)

        return gradient_input

    def _calculate_gradient_bias_3d(self, error_tensor):
        self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

    def _calculate_gradient_weights_1d(self, error_tensor):
        temp_gradient_weights = np.zeros((error_tensor.shape[0], *self.weights.shape))

        padded_input = self._pad_input_1d()

        for batch in range(error_tensor.shape[0]):
            for out_ch in range(error_tensor.shape[1]):
                upsampled_error = self._upsample_error_1d(error_tensor[batch, out_ch])

                for in_ch in range(self.input_tensor.shape[1]):
                    temp_gradient_weights[batch, out_ch, in_ch] = correlate(
                        padded_input[batch, in_ch], upsampled_error, mode='valid')

        self.gradient_weights = temp_gradient_weights.sum(axis=0)

    def _pad_input_1d(self):
        conv_plane_out = []
        for batch in range(self.input_tensor.shape[0]):
            ch_conv_out = []
            for out_ch in range(self.input_tensor.shape[1]):
                padded_channel = np.pad(
                    self.input_tensor[batch, out_ch],
                    (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                    mode='constant')

                if self.convolution_shape[1] % 2 == 0:
                    padded_channel = padded_channel[:-1]

                ch_conv_out.append(padded_channel)

            conv_plane = np.stack(ch_conv_out, axis=0)
            conv_plane_out.append(conv_plane)

        return np.stack(conv_plane_out, axis=0)

    def _upsample_error_1d(self, error):
        temp = resample(error, error.shape[0] * self.stride_shape[0], axis=0)
        temp = temp[:self.input_tensor.shape[2]]

        if self.stride_shape[0] > 1:
            temp[1::self.stride_shape[0]] = 0

        return temp

    def _calculate_gradient_input_1d(self, error_tensor, new_weights, gradient_input):
        for batch in range(error_tensor.shape[0]):
            for out_ch in range(new_weights.shape[0]):
                ch_conv_out = []
                for in_ch in range(new_weights.shape[1]):
                    upsampled_error = self._upsample_error_1d(error_tensor[batch, in_ch])
                    convolved = convolve(
                        upsampled_error, new_weights[out_ch, in_ch], mode='same', method='direct')
                    ch_conv_out.append(convolved)

                gradient_input[batch, out_ch] = np.sum(ch_conv_out, axis=0)

        return gradient_input

    def _calculate_gradient_bias_1d(self, error_tensor):
        self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

    def _rearrange_weights(self, weights, dims):
        if dims == 4:  # 3D Convolution
            return np.transpose(weights, (1, 0, 2, 3))
        elif dims == 3:  # 1D Convolution
            return np.transpose(weights, (1, 0, 2))

    def _update_weights_and_biases(self):
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

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