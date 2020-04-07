from Perceptron import Perceptron


class Predictor:
    """
    Branch Predictor class for perceptron branch predictor simulation
    """
    def __init__(self, n):
        """
        Create branch predictor
        :param n: number of perceptrons
        """
        self.perceptrons = [Perceptron() for i in range(n)]

    def predict(self, n):
        """
        Get prediction from n-th perceptron
        :param n: hash value of address
        :return: 1 if branch is predicted taken; -1 otherwise
        """
        return self.perceptrons[n].predict()

    def train(self, n, x):
        """
        Update perceptron
        :param n: hash value of address
        :param x: true taken/not-taken value
        """
        self.perceptrons[n].train(x)
