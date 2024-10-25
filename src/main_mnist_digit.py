import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore
from DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork
from ReLU import ReLU
from Softmax import Softmax
from src.CrossEntropyLoss import CrossEntropyLoss

# Function to one-hot encode the labels
def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

# Function to predict the class labels using the neural network
def predict(network, X_test):
    predictions = network.forward(X_test)
    return np.argmax(predictions, axis=1)

def evaluate_accuracy(network, X_test, y_test):
    predictions = predict(network, X_test)
    
    accuracy = np.mean(predictions == y_test)
    return accuracy

def main():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0  # Normalize
    X_test = X_test / 255.0

    # Reshape the training and test data to 1D vectors of 784 features (28*28 pixels)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One hot encode training labels
    y_train_encoded = one_hot_encode(y_train, 10)

    nn = NeuralNetwork()

    # Neural network structure
    nn.add_layer(DenseLayer(784, 256))  # Input layer (784 input features for MNIST) to hidden layer
    nn.add_layer(ReLU()) # ReLU activation for hidden layer
    nn.add_layer(DenseLayer(256, 128))
    nn.add_layer(ReLU())  
    nn.add_layer(DenseLayer(128, 10))  # Hidden layer to output layer (10 classes)
    nn.add_layer(Softmax())  # Softmax activation for the output layer

    # Loss = Cross Entropy
    nn.set_loss(CrossEntropyLoss())

    nn.train(X_train, y_train_encoded, epochs=100)

    accuracy = evaluate_accuracy(nn, X_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
