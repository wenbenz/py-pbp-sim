from Predictor import Predictor
from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork
import sys


"""
Evaluates perceptron branch predictor on given an input file path and N, the number of perceptrons
The input file should be as follows:
    - each line represents a unique tuple of (address, taken/not taken)
    - each line should contain 2 numbers separated by a comma
    - taken and not taken should be represented as 1 and 0 respectively
    - example test.txt
"""

if __name__ == '__main__':
    # Input file path
    filepath = sys.argv[1]
    # predictor type; one of {"perceptron", "nn"}
    pred_type = sys.argv[2]
    # Number of predictors
    N = int(sys.argv[3])

    # initialize
    pred_class = Perceptron if str(pred_type).lower() == "perceptron" else NeuralNetwork
    p = Predictor(pred_class, N)
    p_taken, p_not_taken = 0, 0
    hits = 0
    total = 0

    # read test
    f = open(filepath, "r")
    lines = f.readlines()

    # main loop
    for l in lines:
        total += 1
        addr, x = l.split(",")
        addr, x = int(addr) % N, int(x) * 2 - 1    # translates 0s and 1s to -1s and 1s
        y = p.predict(addr)
        if y == 1:
            p_taken += 1
        else:
            p_not_taken += 1
        if x == y:
            hits += 1
        p.train(addr, x)

    # print stats
    print("accuracy:", hits / total)
    print("predicted taken:", p_taken)
    print("predicted not taken:", p_not_taken)
