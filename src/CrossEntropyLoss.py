import numpy as np

class CrossEntropyLoss:
    def forward(self, y_true, y_pred):
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        # Loss = sum of -log(predictions) where y_true is 1
        correct_confidences = np.sum(y_true * -np.log(y_pred_clipped), axis=1)
        return np.mean(correct_confidences)

    def backward(self, y_true, y_pred):
        samples = len(y_true)
        # Gradient of the loss with respect to y_pred
        return (y_pred - y_true) / samples