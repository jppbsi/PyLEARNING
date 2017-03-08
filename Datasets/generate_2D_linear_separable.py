import matplotlib.pyplot as plt

from sklearn.datasets import make_regression

n_samples = 100

plt.title("Synthetic Dataset", fontsize='small')
X, Y = make_regression(n_samples = n_samples, n_features = 1, n_informative = 1, noise = 10, random_state = 0)
plt.scatter(X, Y, color='blue', marker='.')
plt.show()