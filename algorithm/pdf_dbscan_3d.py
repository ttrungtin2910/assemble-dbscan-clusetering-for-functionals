
# Import libraries
import numpy as np
from typing import List
from matplotlib import cm
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Dbscan_3D:
    """
    Methods
    -------
    ```
    - create_dataset_random(self, *args, **kwargs) -> Tuple(np.ndarray)
    - run_algorithm(self) -> None
    - visualize_inference(self) -> None
    """
    def __init__(self):
        ...
    def create_dataset_random(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        visualize: bool = False,
    ):
        """
        Create a dataset of probability density functions (PDFs) based on
        generated data points.

        Parameters
        ----------
        grid_x : ndarray
            A 2D array representing the grid of x-coordinates for evaluating
            the PDFs.
        grid_y : ndarray
            A 2D array representing the grid of y-coordinates for evaluating
            the PDFs.
        visualize : bool, optional
            If set to True, plots the contour of the generated PDFs
            (default is False).

        Returns
        -------
        tuple
            A tuple containing:
            - fi (list of ndarray): A list of PDF values evaluated over the
            grid defined by x and y.
            - label_true (list of int): A list of true labels corresponding
            to each data point.
        """
        # Means and covariances
        mu1 = [1.4, -2]
        mu2 = [-1.5, -2]
        mu3 = [0, 2.4]
        sigma1 = np.array([[0.3, 0], [0, 0.3]])
        sigma2 = np.array([[0.1, 0], [0, 0.3]])
        sigma3 = np.array([[0.2, 0.1], [0.2, 0.3]])

        # Generating data points
        data1 = np.random.multivariate_normal(mu1, sigma1, 50)
        data2 = np.random.multivariate_normal(mu2, sigma2, 50)
        data3 = np.random.multivariate_normal(mu3, sigma3, 50)


        mu = np.vstack((data1, data2, data3))

        label_true = [0]*50 + [1]*50 + [2]*50

        # Initialize variables for contour plotting
        fi = []
        sig = []
        num_sample = mu.shape[0]

        for j in range(num_sample):
            sig_j = 1*np.random.uniform(0.5, 1.5)#3 + 5 * np.random.rand()
            rv = multivariate_normal(mean=mu[j], cov=np.eye(2) / sig_j)
            fi_j = rv.pdf(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
            # Check if normalization
            fi.append(fi_j)
        # Plotting the contours
        if visualize:
            plt.figure()
            for fi_j in fi:
                plt.contour(grid_x, grid_y, fi_j)
            plt.title('Initial Data Points Contours')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.show()

        return fi, label_true

    def run_algorithm(
        self,
        data: List[np.ndarray],
        epsilon: float,
        min_points: int,
        distance: Callable,
        step: float = 0.08
    ) -> np.ndarray:

        """
        Performs DBSCAN clustering on a dataset using a overlap distance matrix.

        This function applies DBSCAN on a dataset `data`, with clusters
        identified based on a neighborhood radius (`epsilon`) and a minimum
        number of points per cluster (`min_points`).
        The distance matrix is calculated using an integration-like similarity
        measure, and noise points are labeled as 0, with clusters starting
        from label 1.

        Parameters
        ----------
        data : list[np.ndarray]
            A list of 2D array where each array is a data sample to be
            clustered.

        epsilon : float
            The neighborhood radius within which points are considered
            neighbors.

        min_points : int
            The minimum number of points required to form a dense region
            (cluster).

        distance: Callable
            Caculation distance function

        step : float
            The incremental step size used in the overlap distance calculation.

        Returns
        -------
        np.ndarray
            A 1D array containing cluster labels for each point in the dataset, 
            with 0 indicating noise points and positive integers denoting clusters.
        """

        # Cluster count
        num_clusters = 0  

        # Initialize DBSCAN results
        num_data = len(data)
        label = np.zeros(num_data)

        # Distance matrix calculation using integration-like similarity measure
        distance_matrix = np.zeros((num_data, num_data))
        for j in range(num_data):
            for i in range(num_data):
                # Using a small constant to avoid zero distances
                distance_matrix[i, j] = distance(
                    data[i],
                    data[j],
                    data_step = step
                )

        # DBSCAN clustering process
        # Track if a point has been visited
        visited = np.zeros(num_data, dtype=bool)
        # Track if a point is classified as noise
        isnoise = np.zeros(num_data, dtype=bool)

        for i in range(num_data):
            if not visited[i]:
                visited[i] = True

                # Find neighbors of point i within epsilon distance 
                neighbors = np.where(distance_matrix[i] <= epsilon)[0]
                #If not enough neighbors, classify as noise
                if len(neighbors) < min_points:
                    isnoise[i] = True
                else:
                    num_clusters += 1
                    label[i] = num_clusters
                    k = 0
                    while True:
                        j = neighbors[k]
                        # Check if point data not belong to any cluster
                        if not visited[j]:
                            visited[j] = True

                            # Get list neighbor
                            neighbors_j = np.where(distance_matrix[j] <= epsilon)[0]
                            if len(neighbors_j) >= min_points:
                                # Add new neighbors
                                for neighbor_temp in neighbors_j:
                                    if neighbor_temp not in neighbors:
                                        neighbors = np.concatenate((neighbors, [neighbor_temp]))

                        if label[j] == 0:
                            # Assign current cluster ID to point j
                            label[j] = num_clusters
                        
                        # Move to the next neighbor
                        k += 1
                        if k >= len(neighbors):
                            break
                    # END while loop
                # END if
            # END if
        # END for loop
        return label
    
    def visualize_inference(
            self,
            f: np.ndarray,
            cluster: np.ndarray,
            grid_x: np.ndarray,
            grid_y: np.ndarray
        ) -> None:
        """
        Plots DBSCAN clustering results in a 3D contour format with distinct
        colors for each cluster.

        This function visualizes clusters identified by DBSCAN by assigning
        unique colors to each cluster.
        Noise points are represented in black. The plot includes a legend
        indicating cluster labels and noise, enhancing interpretability.

        Parameters
        ----------
        f : np.ndarray
            A 3D array containing data points to be plotted, with each
            element representing a 2D contour for a cluster.
        cluster : np.ndarray
            A 1D array of cluster labels, where noise points are labeled as 0,
            and other integers represent specific clusters.
        grid_x : np.ndarray
            A 1D array of x-axis coordinates used for the contour plot.
        grid_y : np.ndarray
            A 1D array of y-axis coordinates used for the contour plot.

        Notes
        -----
        - Each cluster is assigned a distinct color from an HSV colormap.
        - A legend is created based on cluster numbers, including a 'Noise'
        label for unclustered points.
        """
        # Plotting the clusters
        k = int(cluster.max())
        n = len(cluster)

        legend_handles=[]

        colors = cm.hsv(np.linspace(0, 1, k+1))

        plt.figure()
        for i in range(0, k + 1):
            cluster_fi = [f[j] for j in range(n) if cluster[j] == i]
            color = colors[i - 1] if i > 0 else [0, 0, 0]
            legend_text = f'Cluster #{i}' if i > 0 else 'Noise'
            
            for f_j in cluster_fi:
                plt.contour(grid_x, grid_y, f_j, colors=[color], linewidths=1.5)
            
            # Append a proxy artist for the legend entry (only add one per cluster)
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=2, label=legend_text)
            )
            
        plt.title('3D Contour Plot for DBSCAN Clusters')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)

        # Add legend with the handles collected
        plt.legend(handles=legend_handles)

        plt.show()