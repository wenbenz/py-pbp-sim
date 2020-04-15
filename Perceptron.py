import math


class Perceptron:
    """
    Perceptron class for branch prediction
    """

    default_history_length = 62

    def __init__(self, history_length=default_history_length):
        """
        Instantiates a Perceptron with default values taken from
         recommendations in "Dynamic Branch Prediction with Perceptrons"
        :param history_length: length of history to keep
        """
        self.history_length = history_length
        self.threshold = math.floor(1.93 * history_length + 14)
        self.history = []
        self.weights = []

    def _predict(self):
        v = 0
        for h, w in zip(self.history, self.weights):
            v += h * w
        return v

    def predict(self):
        """
        Get perceptron branch prediction
        :return: 1 if branch predicted taken; -1 otherwise
        """
        return int(self._predict() >= 0) * 2 - 1

    def update(self, x):
        """
        Updates the history of the perceptron
        :param x: true taken/not-taken value to add to history
        """
        self.history = [1] + (self.history + [x])[-min(self.history_length, len(self.history) + 1):]
        if len(self.weights) < len(self.history):
            self.weights += [0]

    def train(self, x):
        """
        Update perceptron
        :param x: true taken/not-taken value
        """
        y = self._predict()
        self.update(x)
        if y != x or abs(y) <= self.threshold:
            for i, w in enumerate(self.weights):
                self.weights[i] = w + (x * self.history[i])
