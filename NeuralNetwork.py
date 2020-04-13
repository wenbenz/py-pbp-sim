from sklearn import neural_network
import sys

_history_length = 62
_max_batch_size = 512


class NeuralNetwork:
    """
    Neural Network class for branch prediction
    """
    def __init__(
            self,
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="sgd",
            alpha=.1,
            learning_rate="constant",
            max_iter=99999,
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
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            shuffle=False,
            max_iter=max_iter)
        # initial fit to define the shape of the NN
        self.history = [0 for _ in range(self.history_length)]
        self.classifier.fit([self.history], [0])

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


if __name__ == '__main__':
    nn = NeuralNetwork()

    # Input file path
    filepath = sys.argv[1]
    # Number of perceptrons
    N = int(sys.argv[2])

    f = open(filepath, "r")
    lines = f.readlines()

    # first fit
    nn.history = [int(x.split(",")[1]) * 2 - 1 for x in lines[:nn.history_length]]
    nn.classifier.fit([nn.history], [int(lines[nn.history_length].split(",")[1])*2-1])

    nLines = len(lines)
    total = 0
    hits = 0
    for l in lines[nn.history_length + 1:]:
        total += 1
        addr, x = l.split(",")
        addr, x = int(addr) % N, int(x) * 2 - 1  # translates 0s and 1s to -1s and 1s

        # predict
        y = nn.predict()
        if y == x:
            hits += 1

        # train
        nn.train(x)
        print("Accuracy:", hits / total, "...", total, "/", nLines)
