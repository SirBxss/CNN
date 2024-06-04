from Layers import *
from scipy import stats
import numpy as np

import time

# def test_xavier_distribution():
#     xavier = Xavier()
#     weights_shape = (1000, 1000)
#     fan_in = 1000
#     fan_out = 1000
#
#     weights = xavier.initialize(weights_shape, fan_in, fan_out)
#
#     # Perform a statistical test to check the normality of the distribution
#     k2, p_value = stats.normaltest(weights.flatten())
#
#     # Check if the p-value indicates a good fit (higher p-value means a better fit)
#     assert p_value > 0.01, f"p_value: {p_value}. Possible reason: Formula for Xavier initialization is not implemented correctly."


def test_conv_layer():
    np.random.seed(1337)
    input_tensor = np.random.random((2, 3, 5, 7))
    conv_layer = Conv.Conv((2, 2), (3, 3, 3), 3)

    # Forward pass
    output_tensor = conv_layer.forward(input_tensor)
    print(f"Output Tensor: \n{output_tensor}\n")

    # Generate random error tensor of the same shape as the output
    error_tensor = np.random.random(output_tensor.shape)

    # Backward pass
    gradient_input = conv_layer.backward(error_tensor)
    print(f"Gradient Input: \n{gradient_input}\n")
    print(f"Gradient Weights: \n{conv_layer.gradient_weights}\n")
    print(f"Gradient Bias: \n{conv_layer.gradient_bias}\n")


def test_analytical_gradients():
    np.random.seed(1337)
    input_tensor = np.random.random((2, 3, 5, 7))
    conv_layer = Conv.Conv((2, 2), (3, 3, 3), 3)

    # Forward pass
    output_tensor = conv_layer.forward(input_tensor)

    # Generate random error tensor of the same shape as the output
    error_tensor = np.random.random(output_tensor.shape)

    # Backward pass
    conv_layer.backward(error_tensor)

    print(f"Analytical Gradient Weights: \n{conv_layer.gradient_weights}\n")
    print(f"Analytical Gradient Bias: \n{conv_layer.gradient_bias}\n")



if __name__ == '__main__':
    # input_shape = (3, 10, 14)
    # batch_size = 2
    #
    # input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=float)
    # input_tensor = input_tensor.reshape(batch_size, *input_shape)
    # output_tensor = _conv1d_forward(1, input_tensor)

    # input_shape = (3, 10, 14)
    # batch_size = 2

    # input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=float)
    # print(input_tensor)
    # input_tensor = input_tensor.reshape(batch_size, *input_shape)
    # print(input_tensor)
    #
    # print(input_tensor.shape)
    # print(input_shape[1:])
    #
    # output_x = (14 + 2 * 3 - 8)
    # t = output_x // 1 + 1
    # print(f"yyyy: {output_x}")
    # print(f"t: {t}")

    # test_conv_layer()
    # test_analytical_gradients()

    a = np.zeros([2, 105])
    print(a)