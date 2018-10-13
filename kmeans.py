import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from scipy.linalg import norm


class Kmeans:

    """
    K-means algorithm implementation

    Attributes:

    :attr data: Data points (N by D numpy array)
    :attr N: Number of data points 
    :attr D: Dimension
    :attr K: Number of clusters

    :attr labels: Label of each data point (numpy vector of size N)
    :attr mu: Centroids (K by D numpy array)
    """

    def __init__(self, data, K):

        """
        Class constructor

        :param data: Unlabelled data (Numpy array)
        :param K: Number of clusters
        """

        self.data = data
        self.N, self.D = self.data.shape
        self.K = K

        self.labels = np.zeros(self.N)
        
        index_mu = np.random.permutation(self.N)[:self.K]
        self.mu = np.take(self.data, index_mu,axis=0)

    def compute_labels(self):

        """
        Compute the labels
        """
        
        compare = np.zeros((self.N, self.K))
        for i in range(self.K):
            compare[:,i]= (self.data[:,0] - self.mu[i,0])**2 + (self.data[:,1] - self.mu[i,1])**2
        self.labels = np.argmin(compare,axis=1)
        

    def compute_centroids(self):

        """
        Compute the centroids
        """

        for i in range(self.K):
            clust = np.take(self.data, np.where(self.labels==i)[0], axis=0)
            self.mu[i] = clust.mean(axis=0)


    def run(self, eps):

        """
        Run the K-means algorithm
        """

        diff = 100        
        while diff > eps : 
            temp = self.mu
            self.compute_labels()
            self.compute_centroids()
            diff = np.sqrt(np.sum((temp - self.mu)**2))
       
    def display_data(self):

        """
        Scatter plot of the data
        """
        plt.figure()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.show()


    def display_clusters(self):

        """
        Scatter plot of the data
        """
        plt.figure()
        colors = rainbow(np.linspace(0, 1, self.K))        
        for k in range(self.K):
            clust = np.take(self.data, np.where(self.labels==k)[0], axis=0)
            plt.scatter(clust[:, 0], clust[:, 1],color=colors[k])
        plt.show()

# ------------------------------
# Run the script
# ------------------------------
if __name__ == '__main__':

    # Load Dataset
    data = np.load("Dataset1.npy")
    # Run K-Means algorithm
    inst = Kmeans(data, 2)
    inst.run(0.01)
    inst.display_clusters()
