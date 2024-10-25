import numpy as np
from DenseLayer import DenseLayer

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.learning_rate = 0.001

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def forward(self, X):
        # Forward pass through all layers
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_output):
        # Backward pass through all layers
        for layer in reversed(self.layers):
            # Check if the layer has a learning rate parameter (i.e., it's a DenseLayer)
            if isinstance(layer, DenseLayer):
                d_output = layer.backward(d_output, self.learning_rate)
            else:
                d_output = layer.backward(d_output)

    def train(self, X_train, y_train, epochs, batch_size=64):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                # Get mini-batch
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)

                # Compute loss
                loss_value = self.loss.forward(y_batch, output)

                # Backward pass
                d_loss = self.loss.backward(y_batch, output)
                self.backward(d_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_value}')