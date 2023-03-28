import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# EMClassifier class with an implementation of the Expectation-Maximization algorithm
class EMClassifier:
    def __init__(self, K=3, max_iter=100, epsilon=1e-4):
        # Constructor to initialize the number of clusters, maximum iterations and tolerance level
        self.K = K
        self.max_iter = max_iter
        self.epsilon = epsilon
        # Initializing the parameters
        self.pi = [1/K] * K
        self.mu = None
        self.sigma = None
        self.L= []
    
    def gaussian(self, x, mean, cov):
        # Compute the gaussian probability density function for the given data point and cluster
        d = len(mean)
        det = cov[0][0]*cov[1][1] - cov[0][1]*cov[1][0]
        if det == 0:
            return 0
        norm_const = 1.0 / math.pow((2*math.pi), d/2) * math.pow(det, 1.0/2)
        x_mu = [x[i] - mean[i] for i in range(d)]
        inv = [[cov[1][1], -cov[0][1]], [-cov[1][0], cov[0][0]]]
        pdf = math.pow(math.e, -0.5 * np.dot(np.dot(x_mu, inv), x_mu)) * norm_const
        return pdf
    
    def L_fn(self, X, pi, mu, sigma):
         # Compute the likelihood for the given data using current parameters
        L = [0] * X.shape[0]
        for i in range(X.shape[0]):
            for k in range(self.K):
                L[i] += pi[k] * self.gaussian(X[i], mu[k], sigma[k])
        return L
    
    def plot_clusters(self, X, r):
        # Plot the clusters using the current parameters
        colors = ['r', 'g', 'b']
        for k in range(self.K):
            plt.scatter([X[i][0] for i in range(len(X)) if r[i] == k], [X[i][1] for i in range(len(X)) if r[i] == k], c=colors[k], alpha=0.5)
            plt.scatter(self.mu[k][0], self.mu[k][1], c='k', s=100, marker='x')
            w, v = np.linalg.eigh(self.sigma[k])
            angle = 180.0 / np.pi * np.arctan2(v[0][1], v[0][0])
            ellipse = Ellipse(xy=self.mu[k], width=2*math.sqrt(w[0]), height=2*math.sqrt(w[1]), angle=angle, facecolor='none', edgecolor='k')
            plt.gca().add_artist(ellipse)
        plt.show()
    
    def expectation(self, X):
        # Compute the responsibility matrix using the current parameters
        ret = [[0] * self.K for i in range(len(X))]
        for i in range(len(X)):
            for k in range(self.K):
                ret[i][k] = self.pi[k] * self.gaussian(X[i], self.mu[k], self.sigma[k])
            norm = sum(ret[i])
            ret[i] = [ret[i][k] / norm for k in range(self.K)]
        return ret
    
    def maximization(self, X, r):
        # Compute the updated parameters using the current responsibility matrix
        n_k = [0] * self.K
        self.pi = [0] * self.K
        self.mu = [[0] * X.shape[1] for _ in range(self.K)]
        self.sigma = [np.zeros((X.shape[1], X.shape[1])) for _ in range(self.K)]
        for i in range(X.shape[0]):
            for k in range(self.K):
                n_k[k] += r[i][k]
                self.mu[k] = [self.mu[k][d] + r[i][k] * X[i][d] for d in range(X.shape[1])]
        for k in range(self.K):
            self.pi[k] = n_k[k] / X.shape[0]
            self.mu[k] = [self.mu[k][d] / n_k[k] for d in range(X.shape[1])]
            for i in range(X.shape[0]):
                diff = [X[i][d] - self.mu[k][d] for d in range(X.shape[1])]
                for j in range(X.shape[1]):
                    for l in range(X.shape[1]):
                        self.sigma[k][j][l] += r[i][k] * diff[j] * diff[l] / n_k[k]
        
    def fit(self, X):
        # Initialize mean and covariance for each cluster randomly
        self.mu = [list(X[i]) for i in np.random.choice(X.shape[0], self.K, replace=False)]
        self.sigma = [np.eye(X.shape[1]) for _ in range(self.K)]
        # Run the EM algorithm for a specified number of iterations
        for i in range(self.max_iter):
             # Calculate log likelihood for each data point given the current parameters
            L = self.L_fn(X, self.pi, self.mu, self.sigma)
            self.L.append(np.sum(np.log(L)))
            # Check if the change in log likelihood is less than the specified threshold
            if i > 0 and abs(self.L[-1] - self.L[-2]) < self.epsilon:
                break
            # E-step: calculate the responsibilities for each cluster for each data point
            r = self.expectation(X)
             # M-step: update the parameters based on the responsibilities
            self.maximization(X, r)
             # Visualize the clusters at each iteration
            self.plot_clusters(X, self.predict(X))
    
    def predict(self, X):
        # Calculate the responsibilities for each cluster for each data point
        r = self.expectation(X)
        # Assign each data point to the cluster with the highest responsibility
        return [np.argmax(r[i]) for i in range(X.shape[0])]

data = np.loadtxt('./DATA/donnees_geyser.txt', delimiter=';')
em = EMClassifier(K=2, max_iter=100, epsilon=1e-4)
# Fit the EM model on the data
em.fit(data)
# Plot the log-likelihood values as a function of iterations
plt.plot(em.L)
plt.show()