import numpy as np
import argparse
import csv
import sys

class Linear_Regression:
    
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Regularized Linear Regression')
    parser.add_argument('-tr','--training_data', help='training dataset', required=True)
    parser.add_argument('-ts','--testing_data', help='testing dataset', required=True)
    parser.add_argument('-lr','--learning_rate', help='learning rate', required=False)
    args = vars(parser.parse_args())
    