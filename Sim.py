from Predictor import Predictor
import sys


"""
Evaluates perceptron branch predictor on given an input file path and N, the number of perceptrons
The input file should be as follows:
    - each line represents a unique tuple of (address, taken/not taken)
    - each line should contain 2 numbers separated by a comma
    - taken and not taken should be represented as 1 and -1 respectively
    - example test.txt
"""
if __name__ == '__main__':
    # Input file path
    filepath = sys.argv[1]
    # Number of perceptrons
    N = sys.argv[2]

    p = Predictor(N)
    hits = 0
    total = 0
    f = open(filepath, "r")
    lines = f.readlines()
    for l in lines:
        total += 1
        addr, x = l.split(",")
        addr, x = int(addr) % 10, int(x)
        y = p.predict(addr)
        if x == y:
            hits += 1
    print(hits / total)
