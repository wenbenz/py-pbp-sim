from Predictor import Predictor
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
    # Number of perceptrons
    N = int(sys.argv[2])

    p = Predictor(N)
    p_taken, p_not_taken = 0, 0
    hits = 0
    total = 0

    f = open(filepath, "r")
    lines = f.readlines()
    for l in lines:
        total += 1
        addr, x = l.split(",")
        addr, x = int(addr) % 10, int(x) * 2 - 1    # translates 0s and 1s to -1s and 1s
        y = p.predict(addr)
        if y == 1:
            p_taken += 1
        else:
            p_not_taken += 1
        if x == y:
            hits += 1
        p.train(addr, x)
    print("accuracy:", hits / total)
    print("predicted taken:", p_taken)
    print("predicted not taken:", p_not_taken)
