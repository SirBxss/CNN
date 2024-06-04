import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.loss = 0.0
        self.prediction_tensor = None

        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):

        print(f"\nEpsilon : \n{self.epsilon}")

        self.prediction_tensor = prediction_tensor

        clipped_prediction = np.clip(prediction_tensor, self.epsilon, 1.0 - self.epsilon)

        loss_per_sample = -np.sum(label_tensor * np.log(clipped_prediction))

        self.loss = np.mean(loss_per_sample)

        print("\nClipped Prediction Tensor:")
        print(clipped_prediction)
        print("\nLoss per Sample:")
        print(loss_per_sample)
        print("\nComputed Loss:")
        print(self.loss)

        return self.loss

    def backward(self, label_tensor):
        clipped_prediction = np.clip(self.prediction_tensor, self.epsilon, 1.0 - self.epsilon)

        error_tensor = -(label_tensor / clipped_prediction)

        # error_tensor = self.prediction_tensor - label_tensor

        print(f"\nerror tensor : \n {error_tensor}")
        return error_tensor
