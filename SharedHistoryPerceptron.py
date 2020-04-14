from Perceptron import Perceptron


class SharedHistoryPerceptron:
    """
    Perceptron class for branch prediction with static history
    """
    history = []

    def __init__(self):
        self.perceptron = Perceptron()
        self.perceptron.history = SharedHistoryPerceptron.history

    def train(self, x):
        self.perceptron.train(x)

    def predict(self):
        return self.perceptron.predict()
