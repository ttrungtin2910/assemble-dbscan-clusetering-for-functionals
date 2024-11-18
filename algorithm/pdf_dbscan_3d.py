
# Import libraries
import os

import numpy as np
from typing import List
from matplotlib import cm
from typing import Callable
from datetime import datetime
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
        from sklearn import datasets
        self.datasets_sklearn = datasets
        self.outdir = 'output'
        self.current_time = datetime.now().strftime('%y%m%d_%H%M%S')

        self.dict_data_type = {
            0: 'random_data',
            1: 'noisy_circles',
            2: 'noisy_moons',
            3: 'varied',
            4: 'aniso',
            5: 'blobs',
            6: 'no_structure'
        }

    def make_random_datapoints(
            self,
            n_samples: int,
            index_datatype: int,
        ):
        data_points_type = self.dict_data_type.get(index_datatype, 'random_data')
        if data_points_type == 'random_data':
            # Means and covariances
            mu1 = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]
            mu2 = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]
            mu3 = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]

            # Split n sample into 3 cluster
            num1 = np.random.randint(0, n_samples + 1)
            num2 = np.random.randint(0, n_samples - num1 + 1)
            num3 = n_samples - num1 - num2  # Số thứ 3 sẽ là phần còn lại

            sigma1 = np.array(
                [
                    [np.random.uniform(0.01, 0.05), 0],
                    [0, np.random.uniform(0.01, 0.05)]
                ])
            
            sigma2 =  np.array(
                [
                    [np.random.uniform(0.01, 0.05), 0],
                    [0, np.random.uniform(0.01, 0.05)]
                ])
            sigma3 =  np.array(
                [
                    [np.random.uniform(0.01, 0.05), 0],
                    [0, np.random.uniform(0.01, 0.05)]
                ])

            # Generating data points
            data1 = np.random.multivariate_normal(mu1, sigma1, num1)
            data2 = np.random.multivariate_normal(mu2, sigma2, num2)
            data3 = np.random.multivariate_normal(mu3, sigma3, num3)

            data = np.vstack((data1, data2, data3))
            label_true = [0]*num1 + [1]*num2 + [2]*num3
        else:
            if data_points_type == 'noisy_circles':
                random_data_by_distribution = self.datasets_sklearn.make_circles(
                    n_samples=n_samples, factor=0.5, noise=0.05
                )

            elif data_points_type == 'noisy_moons':
                random_data_by_distribution = self.datasets_sklearn.make_moons(
                    n_samples=n_samples, noise=0.05
                )
                
            elif data_points_type == 'varied':
                random_data_by_distribution = self.datasets_sklearn.make_blobs(
                    n_samples=n_samples,
                    cluster_std=[0.1, 0.3, 0.25],
                    center_box = (-2, 2)
                )
            
            elif data_points_type == 'aniso':
                X, y = self.datasets_sklearn.make_blobs(
                    n_samples=n_samples,
                    center_box = (-2, 2)
                    )
                transformation = [[0.6, -0.6], [-0.4, 0.8]]
                X_aniso = np.dot(X, transformation)
                random_data_by_distribution = (X_aniso, y)

            elif data_points_type == 'blobs':
                random_data_by_distribution = self.datasets_sklearn.make_blobs(
                    n_samples=n_samples,
                    center_box = (-2, 2)
                )

            elif data_points_type == 'no_structure':
                rng = np.random.RandomState(200)
                random_data_by_distribution = rng.rand(n_samples, 2), None

            (data, label_true) = random_data_by_distribution
        
        return data, label_true

    def make_contours(
            self,
            data: np.ndarray,
            grid_x: np.ndarray,
            grid_y: np.ndarray,
        ):
        """
        Create a list of probability density functions (PDFs) for each data 
        point based on a generated covariance matrix and evaluate them over a 
        specified grid of x and y coordinates.

        Parameters
        ----------
        data : ndarray
            A 2D array where each row represents a data point with x and y 
            coordinates.
        grid_x : ndarray
            A 2D array representing the grid of x-coordinates for evaluating 
            the PDFs.
        grid_y : ndarray
            A 2D array representing the grid of y-coordinates for evaluating 
            the PDFs.

        Returns
        -------
        list of ndarray
            A list of PDF values evaluated over the grid defined by x and y.
        """
        # Get the number of data points
        n_samples = data.shape[0]

        fi = []  # List to store the PDF values for each data point

        # Loop through each data point
        for i in range(n_samples):
            # Generate random scaling factors for the covariance matrix (x and y)
            sig_j_x = 50 * np.random.uniform(0.7, 1.3)
            sig_j_y = 50 * np.random.uniform(0.7, 1.3)

            # Create the covariance matrix with the random scaling factors
            cov_matrix = np.array([
                [1 / sig_j_x, 0],
                [0, 1 / sig_j_y]
            ])

            # Create a multivariate normal distribution for each data point
            rv = multivariate_normal(mean=[data[:, 0][i], data[:, 1][i]], 
                                    cov=cov_matrix)

            # Evaluate the PDF over the grid and reshape to match the grid shape
            fi_j = rv.pdf(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)

            # Append the evaluated PDF to the list
            fi.append(fi_j)
        
        # Return the list of PDFs
        return fi

        
    def visualize_raw_data(self, raw_data, f_contours, grid_x, grid_y):
        """
        Visualize the initial raw data points and overlay the contours of the 
        probability density functions (PDFs) for each data point on a 2D grid.

        Parameters
        ----------
        raw_data : ndarray
            A 2D array of raw data points, where each row contains the x and y 
            coordinates.
        f_contours : list of ndarray
            A list of PDF values (contour levels) for each data point, generated 
            by the `make_contours` function.
        grid_x : ndarray
            A 2D array representing the grid of x-coordinates for plotting the 
            contours.
        grid_y : ndarray
            A 2D array representing the grid of y-coordinates for plotting the 
            contours.

        Returns
        -------
        None
            This function does not return anything. It visualizes the data using 
            `matplotlib`.
        """
        # Make directory
        os.makedirs(self.outdir, exist_ok=True)

        filename = os.path.join(
            self.outdir,
            f"{self.current_time}_raw_data.png"
        )

        # Create a new figure for plotting
        plt.figure()

        # Loop through each set of contours (PDFs) and plot them
        for fi_j in f_contours:
            # Plot the contour levels (at 5, 7, and 10) with transparency (alpha=0.5)
            plt.contour(grid_x, grid_y, fi_j, levels=[5, 7, 10], alpha=0.5)

        # Plot the raw data points as red dots
        plt.scatter(raw_data[:, 0], raw_data[:, 1], s=10, color="r")

        # Set the title and labels for the plot
        plt.title('Initial Data Points Contours')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Save the plot to the specified file
        plt.savefig(filename, dpi = 300)  # Save the plot as an image file

        # Optionally close the plot to free up memory
        plt.close()

    @staticmethod
    def add_padding(
            min: float,
            max: float,
            ratio: float = 0.2
        ):
        """
        Add padding to a given range (min, max) by expanding both ends of the range
        by a specified ratio.

        This function calculates the amount of padding based on the difference between
        `max` and `min` values and adds the padding equally to both ends of the range.
        
        Parameters
        ----------
        min : float
            The minimum value of the range.
        max : float
            The maximum value of the range.
        ratio : float, optional
            The padding ratio to apply to the range. The default is 0.2 (20% padding).
            The padding is applied equally to both the lower and upper bounds of the range.

        Returns
        -------
        tuple of floats
            A tuple containing the new `min` and `max` values after padding.
            The first element is the new minimum value, and the second element is
            the new maximum value.
        """
        padding_value = (max - min) * ratio / 2
        return min - padding_value, max + padding_value


    def create_dataset(
            self,
            n_samples: int,
            step: float,
            index_datatype: int = 0,
            visualize: bool = False,
        ):
        data, label_true = self.make_random_datapoints(
            n_samples=n_samples,
            index_datatype=index_datatype
        )
        padding_ratio = 0.2
        
        min_x, max_x = self.add_padding(
            min = data[:,0].min(),
            max = data[:,0].max(),
            ratio = padding_ratio
        )

        min_y, max_y = self.add_padding(
            min = data[:,1].min(),
            max = data[:,1].max(),
            ratio = padding_ratio
        )

        # Create grid x and grid y
        grid_x, grid_y = np.meshgrid(
            np.arange(min_x, max_x + step, step),
            np.arange(min_y, max_y + step, step)
        )

        # Make counters
        f_contours = self.make_contours(
            data=data,
            grid_x=grid_x,
            grid_y=grid_y
        )

        # Plotting the contours
        if visualize:
            self.visualize_raw_data(
                raw_data = data,
                f_contours = f_contours,
                grid_x = grid_x,
                grid_y = grid_y
            )

        return data, f_contours, label_true, grid_x, grid_y

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
            grid_y: np.ndarray,
            point_data: np.ndarray = None,
        ) -> None:
        """
        Plots DBSCAN clustering results in a 3D contour format with distinct
        colors for each cluster. Saves the plot to a file.

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
        point_data : np.ndarray, optional
            If provided, scatter points will be plotted in black on top of the 
            contours.

        Notes
        -----
        - Each cluster is assigned a distinct color from an HSV colormap.
        - A legend is created based on cluster numbers, including a 'Noise'
        label for unclustered points.
        """
        # Make directory
        os.makedirs(self.outdir, exist_ok=True)

        filename = os.path.join(
            self.outdir,
            f"{self.current_time}_cluster_result.png"
        )

        # Plotting the clusters
        k = int(cluster.max())
        n = len(cluster)

        legend_handles = []

        colors = cm.hsv(np.linspace(0, 1, k + 1))

        plt.figure()
        for i in range(0, k + 1):
            cluster_fi = [f[j] for j in range(n) if cluster[j] == i]
            color = colors[i - 1] if i > 0 else [0, 0, 0]
            legend_text = f'Cluster #{i}' if i > 0 else 'Noise'
            
            for f_j in cluster_fi:
                plt.contour(grid_x, grid_y, f_j, colors=[color], levels=[5, 7, 10], 
                            linewidths=1.5, alpha=0.1)
            
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

        if point_data is not None:
            plt.scatter(point_data[:, 0], point_data[:, 1], s=10, color="black", zorder=5)

        # Save the plot to the specified file
        plt.savefig(filename, dpi = 300)  # Save the plot as an image file

        # Close the plot to free up memory
        plt.close()