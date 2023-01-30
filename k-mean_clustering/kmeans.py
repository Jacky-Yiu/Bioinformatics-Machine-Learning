import numpy as np
import matplotlib.pyplot as plt

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        clustering = self.assign_cluster(X)

        for i in range(self.max_iter):
            self.update_centroids(clustering,X)
            clustering = self.assign_cluster(X)
            
        return clustering

    def assign_cluster(self, X: np.ndarray):
        """
        update the cluster base on the curretn centrolds
        """
        clustering = np.zeros(X.shape[0])
        dist = self.euclidean_distance(X, self.centroids)

        for i in range(X.shape[0]):
            clustering[i]= np.argmin(dist[i])

        return clustering 

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        """
        calulate the mean vector of each cluster and replace the old one with the mean vector
        """
        for i in range(len(self.centroids)):
            #find all the row index of that cluster
            indices = np.asarray(clustering==i).nonzero()

            #subset X using the index list
            current_cluster = X[indices]

            #calulate the mean vector
            new_centroid = current_cluster.mean(axis=0)

            #replace the old centrold with the mean vector
            self.centroids[i] = new_centroid


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X: 
        :return:
        """
        print(f"initializing {self.n_clusters} centrolds using {self.init}")
        if self.init == 'random':
            #randomly choose n = self.n_clusters data point
            self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]

        elif self.init == 'kmeans++':
            self.centroids = []

            #using a set to avoid repeated data point
            centroid_index = set()

            #choose the first index randomly
            current_index = np.random.choice(X.shape[0], size=1, replace=False)[0]
            centroid_index.add(current_index)
            self.centroids.append(X[current_index])

            #keep calling select_next_centroid_index until reaching the desired number
            while len(centroid_index)<self.n_clusters:
                current_index = self.select_next_centroid_index(X,X[current_index])

                while current_index in centroid_index:
                    current_index = self.select_next_centroid_index(X,X[current_index])

                centroid_index.add(current_index)
                self.centroids.append(X[current_index])
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # Calulate using numpy L2 norm function
        return np.array([[np.linalg.norm(i-j) for j in X2] for i in X1])

    def select_next_centroid_index(self, X:np.ndarray, y:np.ndarray):
        #get the euclidean_distance using X and all the current centroid
        dist = self.euclidean_distance(X, self.centroids)
        min_distance=[]
        for x in dist:
            #select the distance to the nearest centroid for each data point
            min_distance.append(min(x))

        #increase the wrigth of the distance to the power of 2
        min_distance = [ x**2 for x in min_distance ]

        #normalize the min distanceand get list of probaility
        probalilities = [float(i)/sum(min_distance) for i in min_distance]
    
        #using the probabililty list, we can use np.random.choice to select the data point accordingly
        return np.random.choice(X.shape[0], size=1,p=probalilities, replace=False)[0]


    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        dist = self.euclidean_distance(X, self.centroids)
        silhouette_scores = []
        for i in range(len(dist)):
            sorted_dist = np.sort(dist[i])

            cluster = int(clustering[i])

            second_best = sorted_dist[0]

            if second_best == dist[i][cluster]:
                second_best = sorted_dist[1]

            silhouette_scores.append((second_best-dist[i][cluster])/max(second_best,dist[i][cluster]))

        return sum(silhouette_scores) / len(silhouette_scores)
