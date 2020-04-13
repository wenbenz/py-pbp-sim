from Perceptron import Perceptron


class Predictor:
    """
    Branch Predictor class for perceptron branch predictor simulation
    """
    def __init__(self, predictor_class, n):
        """
        Create branch predictor
        :param predictor_class: class of predictor
        :param n: number of perceptrons
        """
        self.predictor_class = predictor_class
        self.predictors = [self.predictor_class() for _ in range(n)]

    def predict(self, n):
        """
        Get prediction from n-th perceptron
        :param n: hash value of address
        :return: 1 if branch is predicted taken; -1 otherwise
        """
        return self.predictors[n].predict()

    def train(self, n, x):
        """
        Update perceptron
        :param n: hash value of address
        :param x: true taken/not-taken value
        """
        self.predictors[n].train(x)
