import numpy as np
from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer, weight_initializer=None, bias_initializer = None):
        self.optimizer = optimizer
        self.loss = []  # List to store loss value for each iteration
        self.layers = []  # List to hold the architecture
        self.data_layer = None  # Placeholder for data layer (input data and labels)
        self.loss_layer = None  # Placeholder for loss layer (special layer providing loss and prediction)
        self.weights_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        # Get input and labels from data layer
        input_tensor, label_tensor = self.data_layer.next()

        # Forward pass through all layers
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # Store the label tensor for backward pass
        self.label_tensor = label_tensor

        output = self.loss_layer.forward(input_tensor, label_tensor)

        return output

    def backward(self):
        # Start the backward pass from the loss layer using the stored label tensor
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate the error tensor through all layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            self.forward()
            self.backward()
            # Calculate and store the loss for the current iteration
            loss_value = self.loss_layer.loss
            self.loss.append(loss_value)

    def test(self, input_tensor):
        # Forward pass through all layers using the input tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor
