import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


class Ncut:

    """
    Normalized cut implementation

    Attributes:

    :attr data: Data points (N by D numpy array)
    :attr N: Number of data points 
    :attr D: Dimension

    :attr W: Adjacency matrix (Similarity matrix)
    :attr Deg: Degree matrix
    :attr L: Laplacian matrix

    :attr partition: Partition vector
    """

    def __init__(self, data):

        """
        Class constructor

        :param data: Unlabelled data (Numpy array)
        """

        self.data = data
        self.N, self.D = self.data.shape

    def compute_adjacency_matrix(self, threshold):

        """
        Compute the adjacency matrix
        """
        self.W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                dist = np.linalg.norm(self.data[i] - self.data[j])               
                if(dist < threshold):
                    lambd = 0.1
                    self.W[i, j] = np.exp(-dist/lambd)
                    self.W[j, i] = np.exp(-dist/lambd)

    def compute_laplacian(self):

        """
        Compute the degree matrix and the graph Laplacian
        """

        self.Deg = np.diag(self.W.sum(axis = 1))
        self.L = self.Deg - self.W

    def compute_partition(self):

        """
        Compute the graph partition
        """

        w, v = eigh(self.L, self.Deg)        
        idx = np.argpartition(w, 2)[1]
        self.partition = (v[:, idx] < 0)

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
        data1 = np.array([self.data[n] for n in np.arange(self.N) if self.partition[n] == 0])
        data2 = np.array([self.data[n] for n in np.arange(self.N) if self.partition[n] == 1])        
        plt.scatter(data1[:, 0], data1[:, 1], c = 'y')
        plt.scatter(data2[:, 0], data2[:, 1], c = 'b')
        plt.show()
             

# ------------------------------
# Run the script
# ------------------------------
if __name__ == '__main__':

    # Load Dataset
    data = np.load("Dataset2.npy")    
    
    # Run normalized cut algorithm  
    inst = Ncut(data)
    inst.compute_adjacency_matrix(1)
    inst.compute_laplacian()
    inst.compute_partition()
    inst.display_clusters()

	

	

	

