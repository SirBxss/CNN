import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        # To avoid numerical instability, subtract the maximum value from each row of the input tensor
        max_input = np.max(input_tensor, axis=1, keepdims=True)
        exp_input = np.exp(input_tensor - max_input)
        sum_exp_input = np.sum(exp_input, axis=1, keepdims=True)

        # Compute the softmax output by normalizing the exponentiated input
        self.output_tensor = exp_input / sum_exp_input

        return self.output_tensor

    def backward(self, error_tensor):
        # Initialize the error tensor for the previous layer
        # error_tensor_previous = np.zeros_like(self.output_tensor)
        #
        # # Iterate over each example
        # for i in range(self.output_tensor.shape[0]):
        #     # Compute the Jacobian matrix for the current softmax output
        #     softmax_output = self.output_tensor[i]
        #     jacobian_matrix = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)
        #
        #     # Compute the error tensor for the previous layer
        #     error_tensor_previous[i] = np.dot(jacobian_matrix, error_tensor[i])
        #
        # return error_tensor_previous

        # -----------------------------------------------------------------------------------------------------------

        if self.output_tensor is None or error_tensor is None:
            raise ValueError("Output tensor or error tensor is None.")

        error_tensor_previous = error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        error_tensor_previous *= self.output_tensor

        return error_tensor_previous

        # -----------------------------------------------------------------------------------------------------------

        # # Calculate the sum of error_tensor multiplied by self.output_tensor along axis=1
        # sum_error = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        #
        # # Print debugging information
        # print(f"Sum of error tensor and output tensor (axis=1): {sum_error}")
        #
        # # Calculate the error tensor for the previous layer
        # error_tensor_previous = error_tensor - sum_error
        #
        # # Print the difference between error_tensor and sum_error
        # print(f"Error tensor previous: {error_tensor_previous}")
        #
        # # Multiply by output tensor
        # error_tensor_previous *= self.output_tensor
        #
        # # Print the final error tensor previous
        # print(f"Final error tensor previous: {error_tensor_previous}")
        #
        # return error_tensor_previous
