
# Import libraries
import os

import numpy as np
from typing import List
from matplotlib import cm
from typing import Callable
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from ensemble.data_tool import create_similarity_matrix
from ensemble.visualization import visualize_matrix_as_graph_with_coordinates

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
        data_points_type: str,
        random_seed: int = 24,
    ):
        """
        Generates random 2D data points for clustering experiments based on the 
        specified data type and configuration.

        Parameters
        ----------
        n_samples : int
            The total number of data points to generate.
        data_points_type : str
            Type of data distribution to generate. Supported values are:
            - `random_data`: Randomly distributed data points split into 3 clusters.
            - `noisy_circles`: Data points arranged in circular clusters with noise.
            - `noisy_moons`: Data points forming crescent-shaped clusters with noise.
            - `varied`: Clusters of varying density and spread.
            - `aniso`: Anisotropically distributed clusters.
            - `blobs`: Gaussian blobs as clusters.
            - `no_structure`: Uniformly distributed random data points.
        random_seed : int, optional, default=24
            Seed for reproducibility of random number generation.

        Returns
        -------
        data : np.ndarray
            A 2D array of shape (n_samples, 2) containing the generated data points.
        label_true : list or np.ndarray
            The true cluster labels for the generated data points. Each label is an 
            integer representing the cluster to which the point belongs.

        Notes
        -----
        - For 'random_data', three clusters are created with random means, 
        covariances, and sample counts.
        - For 'aniso', an anisotropic transformation is applied to the data points.
        - If `label_true` is None, the data has no specific structure or clustering.
        """
        # data_points_type = self.dict_data_type.get(index_datatype, 'random_data')
        if data_points_type == 'random_data':
            # Means and covariances
            mu1 = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]
            mu2 = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]
            mu3 = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]

            # Split n sample into 3 clusters
            num1 = np.random.randint(0, n_samples + 1)
            num2 = np.random.randint(0, n_samples - num1 + 1)
            num3 = n_samples - num1 - num2

            sigma1 = np.array([[np.random.uniform(0.01, 0.05), 0],
                            [0, np.random.uniform(0.01, 0.05)]])
            sigma2 = np.array([[np.random.uniform(0.01, 0.05), 0],
                            [0, np.random.uniform(0.01, 0.05)]])
            sigma3 = np.array([[np.random.uniform(0.01, 0.05), 0],
                            [0, np.random.uniform(0.01, 0.05)]])

            # Generating data points
            data1 = np.random.multivariate_normal(mu1, sigma1, num1)
            data2 = np.random.multivariate_normal(mu2, sigma2, num2)
            data3 = np.random.multivariate_normal(mu3, sigma3, num3)

            data = np.vstack((data1, data2, data3))
            label_true = [0] * num1 + [1] * num2 + [2] * num3
        else:
            if data_points_type == 'noisy_circles':
                random_data_by_distribution = self.datasets_sklearn.make_circles(
                    n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_seed
                )
            elif data_points_type == 'noisy_moons':
                random_data_by_distribution = self.datasets_sklearn.make_moons(
                    n_samples=n_samples, noise=0.05, random_state=random_seed
                )
            elif data_points_type == 'varied':
                random_data_by_distribution = self.datasets_sklearn.make_blobs(
                    n_samples=n_samples,
                    cluster_std=[0.1, 0.3, 0.25],
                    center_box=(-2, 2),
                    random_state=random_seed
                )
            elif data_points_type == 'aniso':
                X, y = self.datasets_sklearn.make_blobs(
                    n_samples=n_samples,
                    cluster_std=0.2,
                    center_box=(-2, 2),
                    random_state=random_seed
                )
                transformation = [[0.6, -0.6], [-0.4, 0.8]]
                X_aniso = np.dot(X, transformation)
                random_data_by_distribution = (X_aniso, y)
            elif data_points_type == 'blobs':
                random_data_by_distribution = self.datasets_sklearn.make_blobs(
                    n_samples=n_samples,
                    center_box=(-2, 2),
                    random_state=random_seed
                )
            elif data_points_type == 'no_structure':
                rng = np.random.RandomState(random_seed)
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

        
    def visualize_raw_data(
            self,
            raw_data: np.ndarray,
            f_contours: np.ndarray,
            grid_x: np.ndarray,
            grid_y: np.ndarray,
            name: str
            ):
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
        plt.title(f'Initial Data Points Contours: {name}')
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
            data_points_type: str,
            random_seed: int,
            visualize: bool = False,
        ):
        data, label_true = self.make_random_datapoints(
            n_samples=n_samples,
            data_points_type=data_points_type,
            random_seed=random_seed
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
                grid_y = grid_y,
                name = f"{data_points_type}_{random_seed}"
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
        name: str = '',
        description: str = '',
        ax=None
    ):
        """
        Visualizes DBSCAN clustering results as 3D contours with distinct colors for
        each cluster. Optionally, scatter points can be plotted on top of the contours.
        
        Parameters
        ----------
        f : np.ndarray
            A 3D array of shape (n, h, w) where each slice `f[j]` contains the 2D 
            contour data for the j-th cluster.
        cluster : np.ndarray
            A 1D array of shape (n,) containing cluster labels for each data point.
            Noise points are labeled as 0, and other integers represent specific clusters.
        grid_x : np.ndarray
            A 1D array representing the x-axis coordinates for the contour plot.
        grid_y : np.ndarray
            A 1D array representing the y-axis coordinates for the contour plot.
        point_data : np.ndarray, optional
            A 2D array of shape (m, 2), where each row represents a scatter point 
            plotted on top of the contours. Default is None.
        name : str, optional
            A string representing the data type being visualized, used in the plot title.
        description : str, optional
            A description string displayed in the title for additional context.
        ax : matplotlib.axes.Axes, optional
            The axis object where the plot will be drawn. If not provided, a new 
            axis will be created.

        Notes
        -----
        - The function assigns unique colors to each cluster using an HSV colormap.
        - Noise points are visualized with black contours.
        - If `point_data` is provided, it is plotted as black scatter points.
        - The function can either draw the plot on the provided axis or save it directly.

        Returns
        -------
        None
            The function modifies the provided `ax` object or creates a standalone
            figure if no axis is provided.
        """
        # If no axis is provided, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots()

        # Determine the number of clusters
        k = int(cluster.max())
        n = len(cluster)

        # Initialize the colormap for clusters
        colors = cm.hsv(np.linspace(0, 1, k + 1))
        legend_handles = []

        # Plot contours for each cluster
        for i in range(0, k + 1):
            # Extract all slices corresponding to cluster i
            cluster_fi = [f[j] for j in range(n) if cluster[j] == i]

            # Set color for the cluster (black for noise)
            color = colors[i - 1] if i > 0 else [0, 0, 0]
            legend_text = f'Cluster #{i}' if i > 0 else 'Noise'

            # Plot contours for this cluster
            for f_j in cluster_fi:
                ax.contour(grid_x, grid_y, f_j, colors=[color], levels=[5, 7, 10, 100],
                        linewidths=1.5, alpha=0.1)

            # Add a legend entry for this cluster
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=2, label=legend_text)
            )

        # Set title, labels, and grid
        ax.set_title(f'{name}\n{description}', fontsize=10)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.grid(True)

        # Add legend
        ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0, 0))

        # If point data is provided, scatter plot the points
        if point_data is not None:
            ax.scatter(point_data[:, 0], point_data[:, 1], s=10, color="black", zorder=5)
    
    def visualize_result_as_graph(
            self,
            list_label_infer: list,
            point_data: np.ndarray,
        ):
        """
        Visualizes clustering results as graph-based representations for each cluster, 
        using similarity matrices and point coordinates.

        Parameters
        ----------
        list_label_infer : list
            A list of cluster labels inferred for each data point. Each element 
            represents the cluster assignment of the corresponding point in `point_data`.
        point_data : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the data points. 
            Each row corresponds to a data point, and columns represent its features.

        Notes
        -----
        - The function creates a grid of subplots where each subplot represents the 
        graph visualization of one cluster.
        - The similarity matrix for each cluster is generated using a helper function 
        `create_similarity_matrix`.
        - Graph visualization is done by `visualize_matrix_as_graph_with_coordinates`, 
        which plots the graph with cluster-specific data points and similarity matrix.
        - Unused subplots in the grid are hidden.
        - The final visualization is saved as an image file in the specified output directory.

        Saves
        -----
        A PNG file named `<timestamp>_cluster_result_graph_visualization.png` is saved 
        in the directory `self.outdir` with high resolution (300 dpi).
        """
        label_infer = list_label_infer[-1]

        # similarity_matrix = create_similarity_matrix(label_infer)
        num_clusters = set(label_infer)

        num_cols = 2  # Number of columns in the subplot grid
        num_rows = (len(num_clusters) + num_cols - 1) // num_cols  # Calculate the number of rows needed

        # Create a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array for easy access

        # Loop through each cluster and plot the corresponding graph
        for i, index_cluster in enumerate(num_clusters):
            # Create a mask for the current cluster
            consider_list = [-1 if element == index_cluster else 0 for element in label_infer]

            # Generate the similarity matrix for the current cluster
            similarity_matrix = create_similarity_matrix(consider_list)

            # Call the visualization function to plot on the corresponding subplot
            visualize_matrix_as_graph_with_coordinates(
                matrix=similarity_matrix,
                list_points=point_data,
                index_cluster=index_cluster,
                ax=axes[i]
            )

        # Hide any unused subplots
        for j in range(len(num_clusters), len(axes)):
            axes[j].axis('off')

        filename = os.path.join(
            self.outdir,
            f"{self.current_time}_cluster_result_graph_visualization.png"
        )

        # Save the plot to the specified file
        plt.savefig(filename, dpi = 300, bbox_inches='tight')  # Save the plot as an image file

        # Close the plot to free up memory
        plt.close()