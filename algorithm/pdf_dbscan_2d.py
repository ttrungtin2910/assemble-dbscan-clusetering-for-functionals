# Import libraries
import numpy as np
from scipy.stats import norm

from matplotlib import cm
import matplotlib.pyplot as plt

from typing import Tuple, Callable

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