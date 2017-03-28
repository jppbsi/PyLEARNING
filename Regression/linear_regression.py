import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import argparse
import csv
import sys

def main():

    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--data', help='dataset', required=True)
    args = vars(parser.parse_args())
    
    data = np.matrix(np.loadtxt(open(sys.argv[2], 'r'), delimiter=" ", skiprows=1))
    
    X = np.array(data[:,1])
    Y = np.array(data[:,0])
    
    model = linear_model.LinearRegression()
    model.fit(X, Y)
    Y_prime = model.predict(X)
    
    plt.scatter(X, Y, color='blue', marker='o')
    plt.plot(X, Y_prime, color='black', linewidth=2)
    plt.show()
    
    print("Mean squared error: %.2f" % np.mean((Y_prime - Y) ** 2))
    
if __name__ == '__main__':
  main()