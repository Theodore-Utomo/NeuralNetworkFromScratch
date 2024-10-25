import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        # Weight initialization (He initialization for ReLU)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input_data = input_data
        # Compute the linear combination of inputs and weights (Z = XW + b)
        self.output = np.dot(self.input_data, self.weights) + self.biases
        return self.output

    def backward(self, d_output, learning_rate):
        # Calculate gradients
        d_weights = np.dot(self.input_data.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.weights.T)
        
        # Gradient descent update
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input