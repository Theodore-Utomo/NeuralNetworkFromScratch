import numpy as np

class ReLU:
    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)
    
    def backward(self, d_output):
        return d_output * (self.input_data > 0)  # ReLU derivative