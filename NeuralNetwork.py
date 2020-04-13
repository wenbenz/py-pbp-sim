from sklearn import neural_network
import random


_history_length = 62
_max_batch_size = 512   # max batch size for fitting the model

_hidden_layer_sizes = (8,)
_activation = "relu"
_solver = "sgd"
_alpha = .125
_learning_rate = "constant"
_max_iter = 99999


class NeuralNetwork:
    """
    Neural Network class for branch prediction
    """
    def __init__(
            self,
            history_length=_history_length):
        """
        Instantiates a Neural Network with default value equal to that
         of the recommendation for perceptron branch predictor in
         "Dynamic Branch Prediction with Perceptrons"
        :param history_length: length of history to keep
        """
        self.history_length = history_length
        self.history = []
        self.batch_X = []
        self.batch_Y = []
        self.batch_size = 1
        self.current_batch_size = 0
        self.classifier = neural_network.MLPClassifier(
            hidden_layer_sizes=_hidden_layer_sizes,
            activation=_activation,
            solver=_solver,
            alpha=_alpha,
            learning_rate=_learning_rate,
            shuffle=False,
            max_iter=_max_iter)
        # initial fit to define the shape of the NN
        self.history = [random.randint(0, 1) for _ in range(self.history_length)]
        self.classifier.fit([self.history], [random.randint(0, 1)])

    def predict(self):
        """
        Get perceptron branch prediction
        :return: 1 if branch predicted taken; -1 otherwise
        """
        return self.classifier.predict([self.history])[0]

    def update(self, x):
        """
        Updates the history of the perceptron
        :param x: true taken/not-taken value to add to history
        """
        self.history = (self.history + [x])[-self.history_length:]

    def train(self, x):
        """
        Update perceptron
        :param x: true taken/not-taken value
        """
        self.update(x)
        self.batch_X += [self.history]
        self.batch_Y += [x]
        self.current_batch_size += 1
        if self.current_batch_size == self.batch_size:
            self.current_batch_size = 0
            self.classifier.fit(self.batch_X, self.batch_Y)
            self.batch_X = []
            self.batch_Y = []
            self.batch_size = min(self.batch_size + 1, _max_batch_size)
