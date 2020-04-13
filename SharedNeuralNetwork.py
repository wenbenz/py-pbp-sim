from NeuralNetwork import NeuralNetwork


class SharedNeuralNetwork:
    """
    Neural Network class for branch prediction with a static classifier.
    Almost the same as NN class, except the classifier (and weights) are
    shared
    """
    classifier = NeuralNetwork().classifier

    def __init__(self):
        self.nn = NeuralNetwork()
        self.nn.classifier = SharedNeuralNetwork.classifier

    def train(self, x):
        self.nn.train(x)

    def predict(self):
        return self.nn.predict()
