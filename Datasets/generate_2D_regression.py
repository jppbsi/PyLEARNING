import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys

from sklearn.datasets import make_regression

def main():
    
    parser = argparse.ArgumentParser(description='2D Synthetic dataset generation for regression problems')
    parser.add_argument('-s','--samples', help='number of samples', required=True)
    parser.add_argument('-n','--noise', help='noise? (yes|no)', required=True)
    parser.add_argument('-o','--output', help='output file name', required=True)
    
    args = vars(parser.parse_args())
    
    n_samples = int(sys.argv[2])
    n_features = 1
    
    plt.title("Synthetic Dataset", fontsize='small')    
    if(sys.argv[4].upper() == "YES"):
        X, Y = make_regression(n_samples = n_samples, n_features = n_features, n_informative = 1, noise = 10, random_state = 0)
    else:
        X, Y = make_regression(n_samples = n_samples, n_features = n_features, n_informative = 1, noise = 0, random_state = 0)
    
    plt.scatter(X, Y, color='blue', marker='o')
    plt.show()
    np.savetxt(sys.argv[6], np.c_[Y, np.array(X)], fmt = '%.4f', header = str(n_samples)+' '+str(n_features), comments = '')

if __name__ == '__main__':
  main()