import numpy as np

class Softmax:
    def forward(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, d_output):
        # Not commonly used in this form, handled with cross-entropy
        return d_output