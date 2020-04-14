from Predictor import Predictor
from Perceptron import Perceptron
from NeuralNetwork import NeuralNetwork
from SharedNeuralNetwork import SharedNeuralNetwork
from SharedHistoryPerceptron import SharedHistoryPerceptron
import sys
import time


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
    pred_type = sys.argv[2].lower()
    # Number of predictors
    N = int(sys.argv[3])

    pred_class = Perceptron
    if pred_type == "nn":
        pred_class = NeuralNetwork
    elif pred_type == "snn":
        pred_class = SharedNeuralNetwork
    elif pred_type == "shp":
        pred_class = SharedHistoryPerceptron

    # initialize
    p = Predictor(pred_class, N)
    p_taken, p_not_taken = 0, 0
    hits = 0
    total = 0
    start_time = time.time()
    t = int(time.time())

    # read test
    f = open(filepath, "r")
    lines = f.readlines()
    nlines = len(lines)

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
        # Print progress
        if t != int(time.time()):
            t = int(time.time())
            print("accuracy:", hits / total, "...", total, "/", nlines, " done... Elapsed time:", int(t - start_time), "s.")

    # print stats
    print("accuracy:", hits / total)
    print("predicted taken:", p_taken)
    print("predicted not taken:", p_not_taken)
