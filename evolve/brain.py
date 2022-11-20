import numpy as np
from .settings import LEARNING_EPOCHS, LEARNING_RATE


class Brain(object):
    def __init__(
        self,
        epochs=LEARNING_EPOCHS,
        learning_rate=LEARNING_RATE,
        n_input=3,
        n_hidden=4,
        n_out=1,
    ):
        # Initialize hyperparameter variables.
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.errors = []

        # Initialize weights and biases.
        self.hidden_weight = np.random.random(size=(self.n_input + 1, self.n_hidden))
        self.output_weight = np.random.random(size=(self.n_hidden + 1, self.n_out))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def fit(self, features, labels):
        for _ in range(self.epochs):
            out = self.predict(features)
            self.backpass(features, labels, out)

    def backpass(self, features, labels, out):
        labels = np.array(labels)
        features = np.array(features)
        error = labels - out

        self.errors.append(np.sum(error ** 2))
        # Calculate adjustment from hidden -> output.
        delta_output = self.sigmoid_prime(out) * error

        # Calculate error from input -> hidden.
        output_error = delta_output.dot(self.output_weight[1:].T)
        delta_hidden = output_error.sum(axis=1).reshape(
            output_error.shape[0], 1
        ) * self.sigmoid_prime(out)

        # Adjust hidden -> output weights.

        self.output_weight[1:] += (
            self.activated_hidden.T.dot(delta_output) * self.learning_rate
        )
        self.output_weight[0] = np.sum(delta_output)
        self.hidden_weight[1:] += (
            features.T.dot(delta_hidden).mean(axis=1).reshape(self.n_input, 1)
            * self.learning_rate
        )
        self.hidden_weight[0] = np.sum(delta_hidden)

    def predict(self, features):
        inputs = np.dot(features, self.hidden_weight[1:]) + self.hidden_weight[0]
        self.activated_hidden = self.sigmoid(inputs)
        output = (
            np.dot(self.activated_hidden, self.output_weight[1:])
            + self.output_weight[0]
        )
        final = self.sigmoid(output)
        return final
