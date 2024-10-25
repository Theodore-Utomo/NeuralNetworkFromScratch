import numpy as np

class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, d_output):
        return d_output * (self.output * (1 - self.output)) 