import math

_history_length = 62
_threshold = math.floor(1.93 * _history_length + 14)


class Perceptron:
    """
    Perceptron class for branch prediction
    """
    def __init__(self, history_length=_history_length, threshold=_threshold):
        """
        Instantiates a Perceptron with default values taken from
         recommendations in "Dynamic Branch Prediction with Perceptrons"
        :param history_length: length of history to keep
        :param threshold: highest absolute weight
        """
        self.history_length = history_length
        self.threshold = threshold
        self.history = []
        self.weights = []

    def predict(self):
        """
        Get perceptron branch prediction
        :return: 1 if branch predicted taken; -1 otherwise
        """
        v = 0
        for h, w in zip(self.history, self.weights):
            v += h * w
        return int(v >= 0) * 2 - 1

    def update(self, x):
        """
        Updates the history of the perceptron
        :param x: true taken/not-taken value to add to history
        """
        self.history = (self.history + [x])[-min(self.history_length, len(self.history) + 1):]
        if len(self.weights) < len(self.history):
            self.weights += [0]

    def train(self, x):
        """
        Update perceptron
        :param x: true taken/not-taken value
        """
        self.update(x)
        y = self.predict()
        if y != x or abs(y) <= self.threshold:
            for i, w in enumerate(self.weights):
                self.weights[i] = w + (x * self.history[i])
