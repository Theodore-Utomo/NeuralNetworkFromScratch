import unittest
import numpy as np
from NeuralNetwork import NeuralNetwork
from DenseLayer import DenseLayer
from ReLU import ReLU
from Softmax import Softmax
from src.CrossEntropyLoss import CrossEntropyLoss
from main_mnist_digit import one_hot_encode, predict, evaluate_accuracy

class TestNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        # Create a basic neural network structure for testing
        self.nn = NeuralNetwork()
        self.nn.add_layer(DenseLayer(784, 10))  # Simple layer for quick test
        self.nn.add_layer(Softmax())
        self.nn.set_loss(CrossEntropyLoss())

        # Mock input data (2 samples, 784 features)
        self.mock_data = np.array([[1.0] * 784, [0.0] * 784])
        
        # Mock one-hot encoded labels for testing
        self.mock_labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    def test_one_hot_encode(self):
        # Test one-hot encoding
        y = np.array([0, 1, 2])
        one_hot_encoded = one_hot_encode(y, 3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(one_hot_encoded, expected, "One-hot encoding failed")

    def test_predict(self):
        # Test the predict function
        self.nn.train(self.mock_data, self.mock_labels, epochs=1)
        predictions = predict(self.nn, self.mock_data)
        self.assertEqual(predictions.shape, (2,), "Predict function output shape mismatch")

    def test_evaluate_accuracy(self):
        # Train for more epochs to give the model a better chance to learn
        self.nn.train(self.mock_data, self.mock_labels, epochs=10)
        accuracy = evaluate_accuracy(self.nn, self.mock_data, np.array([0, 1]))
        
        # Now pass the message correctly with the 'msg' argument
        self.assertAlmostEqual(accuracy, 1.0, places=5, msg="Accuracy evaluation failed")

    
    def test_forward_pass(self):
        # Test the forward pass to ensure layers work correctly
        output = self.nn.forward(self.mock_data)
        self.assertEqual(output.shape, (2, 10), "Forward pass output shape mismatch")

    def test_cross_entropy_loss(self):
        # Test the cross-entropy loss calculation
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(y_true, y_pred)
        self.assertGreaterEqual(loss, 0, "Cross-entropy loss should be non-negative")

if __name__ == '__main__':
    unittest.main()
