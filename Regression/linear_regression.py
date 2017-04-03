#TO DO: to implement multivariate linear regression
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import argparse
import csv
import sys

class Linear_Regression:
    
    def __init__(self, num_variables, num_samples, learning_rate = 0.1):
        self.learning_rate = learning_rate
        self.n = num_variables
        self.m = num_samples
        
        # It initializes a weight array of dimension (n+1) using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.w = 0.1*np.random.randn(self.n+1)  #It creates w = [w_0 w_1 ... w_n], where n stands for the number of decision variables. It
        
    
    def train(self, X, Y):
        """
        It trains (learns the hypothesis function) the linear regressor
    
        Parameters
        ----------
        X: input array with the dataset
        Y: desired outputs for the data
    
        """
        Y_prime = np.zeros(self.m)
        
        old_error = 0.0
        current_error = 1.0
        print self.w
        
        while np.absolute(current_error-old_error) > 0.001:
            old_error = current_error
            
            error_0 = 0.0
            error_1 = 0.0
            for i in range(0, self.m):
                error_0 = error_0 + (self.h_w(X[i]) - Y[i])
                error_1 = error_1 + (self.h_w(X[i]) - Y[i])*X[i]
            error_0 = error_0/self.m
            error_1 = error_1/self.m
   
            # Updating the weights
            self.w[0] = self.w[0] - self.learning_rate*error_0
            self.w[1] = self.w[1] - self.learning_rate*error_1
        
            #Executing the hypothesis function once more to compute the updated error
            for i in range(0, self.m):
                Y_prime[i] = self.h_w(X[i])
            current_error = np.mean((Y_prime - Y) ** 2)
            print("Mean squared error: %.2f" % np.mean((Y_prime - Y) ** 2))
            
    
    def h_w(self, x):
        """
        It computes a linear hypothesis function: h_w(x) = w_0 + w_1*x.
    
        Parameters
        ----------
        x: input array with num_variables dimensions
    
        Returns
        -------
        h_prime: output of the hypothesis function
        """
        h_prime = self.w[0] + self.w[1]*x
        
        return h_prime

def main():

    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--data', help='dataset', required=True)
    parser.add_argument('-l','--learning_rate', help='learning rate', required=True)
    args = vars(parser.parse_args())
    
    data = np.matrix(np.loadtxt(open(sys.argv[2], 'r'), delimiter=" ", skiprows=1))
    
    X = np.array(data[:,1])
    Y = np.array(data[:,0])
    
    num_samples = X.shape[0]
    
    r = Linear_Regression(1, num_samples, learning_rate = float(sys.argv[4]))
    r.train(X, Y)
    
    #model = linear_model.LinearRegression()
    #model.fit(X, Y)
    #Y_prime = model.predict(X)
    
    #plt.scatter(X, Y, color='blue', marker='o')
    #plt.plot(X, Y_prime, color='black', linewidth=2)
    #plt.show()
    
    #print("Mean squared error: %.2f" % np.mean((Y_prime - Y) ** 2))
    
if __name__ == '__main__':
  main()