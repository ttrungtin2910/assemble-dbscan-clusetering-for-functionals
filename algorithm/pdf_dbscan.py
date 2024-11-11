# Import libraries
import numpy as np
from typing import List
from matplotlib import cm
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from scipy.stats import multivariate_normal

class Dbscan2D:
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
            grid: np.ndarray = None,
            abnormal_params: list = None,
            visualize: bool = False
        ) -> Tuple[np.ndarray]:

        """
        Create dataset fx, fy by defauls
        Generate probability density functions (PDFs) and its labels used for
        clustering algorithm

        Parameters
        ----------
        grid: np.ndarray
            A NumPy array representing the points over which the PDFs are
            evaluated.

        abnormal_params : list of lists, optional
            A list of abnormal distribution parameters, where each element
            is a pair of:
            - `mus`: List of means for the abnormal group.
            - `sigmas`: List of corresponding standard deviations for eachmean.
            If provided, the function will generate PDFs for these abnormal
            distributions.
        
        visualize: bool
            If true, the function will visualize the data

        Returns
        -------
        data : np.ndarray
            A 2D NumPy array - each column is the PDF of a distribution,
            with rows representing values evaluated over the `grid`.

        labels : np.ndarray
            A 1D NumPy array of labels for each PDF:
            - Regular group distributions are labeled starting from 1.
            - Abnormal distributions are assigned the label `num_groups + 1`.
        """
        # Check if grid is not existed
        if grid is not None:
            # Declare param input
            start = -50
            stop = 50
            step = 0.2

            # Create range data
            grid = np.arange(start, stop + step, step)
        # 
        
        # Generate random means for the PDFs
        mu1 = np.random.normal(10, 5, 20)
        mu2 = np.random.normal(-10, 2, 5)

        # Merge as a list 
        mu_ranges = [mu1, mu2]

        # Create rundom sig
        sig_values = [6, 9]

        # Initialize variables
        num_groups = len(mu_ranges)
        pdfs = []
        labels = []

        # Generate PDFs for each group
        for group_index, mu_range in enumerate(mu_ranges):
            for mu in mu_range:
                f_single = norm.pdf(
                    grid,
                    loc=mu,
                    scale=sig_values[group_index]
                )
                pdfs.append(f_single)
                labels.append(group_index + 1)  # MATLAB indices start at 1

        # Generate PDFs for abnormal distributions
        if abnormal_params is not None:
            for abnormal_group in abnormal_params:
                if isinstance(abnormal_group, list)\
                        and len(abnormal_group) == 2:
                    mus = abnormal_group[0]
                    sigmas = abnormal_group[1]
                    abnormal_pdf = np.zeros_like(grid)
                    for mu, sigma in zip(mus, sigmas):
                        abnormal_pdf += norm.pdf(grid, loc=mu, scale=sigma)
                    pdfs.append(abnormal_pdf)
                    # Assign label for "abnormal" group
                    labels.append(num_groups + 1)

        # Convert lists to NumPy arrays
        data = np.array(pdfs).T  # Transpose to match MATLAB's orientation
        labels = np.array(labels)

        # Plot the original PDFs in gray color
        if visualize:
            plt.figure(figsize=(12, 6))
            for i in range(data.shape[1]):
                plt.plot(grid, data[:, i], color=[0.8, 0.8, 0.8])
            plt.title('Original PDFs')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.grid(True)
            plt.show()
        
        # Return value
        return data, labels


    def run_algorithm(
        self,
        data: np.ndarray,
        epsilon: float,
        min_points: int,
        distance: Callable,
        step: float = 0.2
    ) -> np.ndarray:
        """
        Applies the DBSCAN algorithm to cluster data points based on density
        and distance criteria.

        The function performs DBSCAN clustering by calculating an L1 distance,
        using each data point's neighbors within a specified radius (`epsilon`)
        and a minimum point threshold (`min_points`). Points that don't meet
        the density requirements are classified as noise.

        Parameters
        ----------
        data : np.ndarray
            A 2D array representing pdf data points. Each element is pdf
            
        epsilon : float
            The radius of the neighborhood for each point
            
        min_points : int
            The minimum number of points required in a neighborhood to
            consider a point is core point

        distance: Callable
            Caculation distance function
            
        step : float
            The step size used for calculating the distance metric.

        Returns
        -------
        np.ndarray
            A 1D array of integer labels where each element corresponds to
            a data point's cluster assignment. Noise points are labeled
            as 0, while other points have positive integer labels indicating
            their cluster membership.
        """

        num_clusters = 0
        # Number of data points
        num_data = data.shape[1]

        # Initialize results for DBSCAN
        labels_infer = np.zeros(num_data, dtype=int)

        # Compute the distance matrix
        distance_matrix = np.zeros((num_data, num_data))

        # Calculate distance matrix
        for j in range(num_data):
            for i in range(num_data):
                # Calculate distance
                distance_matrix[i, j] = distance(
                    data1=data[:, i],
                    data2=data[:, j],
                    data_step=step,
                    num_dim=1
                ) 

        # Create temp variables
        # Track if a point has been visited
        visited = np.zeros(num_data, dtype=bool)
        # Track if a point is classified as noise
        isnoise = np.zeros(num_data, dtype=bool)   

        # DBSCAN clustering process
        for i in range(num_data):
            if not visited[i]:
                visited[i] = True

                # Find neighbors of point i within epsilon distance 
                neighbors = np.where(distance_matrix[i, :] <= epsilon)[0]
                #If not enough neighbors, classify as noise
                if len(neighbors) < min_points:
                    isnoise[i] = True
                else:
                    num_clusters += 1
                    # Assign cluster label to point i
                    labels_infer[i] = num_clusters

                    # Expand cluster (Mở rộng cụm)
                    k = 0
                    while True:
                        j = neighbors[k]
                        
                        # Check if point data not belong to any cluster
                        if not visited[j]:
                            visited[j] = True

                            # Get list neighbor
                            neighbors_j = np.where(distance_matrix[j, :] <= epsilon)[0]
                            if len(neighbors_j) >= min_points:

                                # Add new neighbors
                                for neighbor_temp in neighbors_j:
                                    if neighbor_temp not in neighbors:
                                        neighbors = np.concatenate((neighbors, [neighbor_temp]))

                        if labels_infer[j] == 0:
                            # Assign current cluster ID to point j
                            labels_infer[j] = num_clusters
                        # PlotDBSCAN(f, IDX, param)
                        # Move to the next neighbor
                        k += 1
                        
                        if k >= len(neighbors):
                            break
                    # END loop while
                # END if else
            # END if else
        # END loop for
        return labels_infer
        

    def visualize_inference(
            self,
            f: np.ndarray,
            cluster: np.ndarray,
            grid: np.ndarray
        ) -> None:

        """
        Plots the results of density-based spatial clustering (DBSCAN) with color-coding for clusters and noise.

        This function visualizes clusters determined by DBSCAN by assigning distinct colors to each cluster, 
        while noise points (label 0) are marked in black. It generates a line plot for each cluster, 
        allowing for a clear visual differentiation of clusters in the dataset.

        Parameters
        ----------
        f : np.ndarray
            A 2D array where each row represents a dimension and each column corresponds to a data point. 
            It stores the feature values to be plotted.
        
        cluster : np.ndarray
            A 1D array of cluster labels assigned to each data point, where noise points are labeled as 0, 
            and other positive integers indicate specific cluster memberships.
        
        grid : np.ndarray
            A 1D array of x-axis values for plotting, representing the indices or feature positions 
            corresponding to each data point in `f`.

        Returns
        -------
        None
            This function only creates and displays the plot, with no return value.

        
        Notes
        -----
        - Each cluster is assigned a unique color, while noise points are always plotted in black.
        - A legend is created based on the clusters, including a label for noise if present.
        - This function uses a HSV colormap for assigning colors to clusters, enhancing visual clarity 
        when there are multiple clusters.
        """

        # Determine the number of clusters (excluding noise labeled as 0)
        k = np.max(cluster)

        # Generate colors for each cluster using HSV colormap
        list_colors = cm.hsv(np.linspace(0, 1, k+1))

        # Prepare lists for legend handles and text entries
        legend = []
        legend_text_entries = []

        # Iterate over clusters (including noise with index 0)
        for i in range(k + 1):
            # Select data points belonging to the current cluster (or noise)
            fi = f[:, cluster == i]

            if i != 0:  # If not noise
                color = list_colors[i - 1]  # Assign color to the cluster
                legendText = f'Cluster #{i}'
            else:  # Noise points
                color = [0, 0, 0]  # Black color for noise
                legendText = 'Noise'

            # Plot only if there are data points in the current group
            if fi.size > 0:
                h = plt.plot(grid, fi, color=color)  # Plot data points
                legend.append(h[0])  # Store plot handle for legend
                legend_text_entries.append(legendText)  # Store legend text

        # Add grid and legend to the plot
        plt.grid(True)
        plt.legend(legend, legend_text_entries, loc='upper right')
        
        # Display the plot
        plt.show()


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