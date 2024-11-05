import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def plot_2d(
        f: np.ndarray,
        cluster: np.ndarray,
        data_points: np.ndarray
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
    
    data_points : np.ndarray
        A 1D array of x-axis values for plotting, representing the indices or feature positions 
        corresponding to each data point in `f`.

    Returns
    -------
    None
        This function only creates and displays the plot, with no return value.

    Example
    -------
    >>> grid = np.arange(-50, 50 + 0.2, 0.2)
    >>> f = np.array([norm.pdf(grid, loc=0+i, scale=1) for i in range(10)]).T  # Example feature array (2D data)
    >>> cluster = np.array([1, 1, 0, 2, 2] * 2)  # Example cluster labels
    # data_points = np.arange(500)  # Example x-axis data points
    >>> plot_fDBSCAN(f, cluster, grid)
    
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
            h = plt.plot(data_points, fi, color=color)  # Plot data points
            legend.append(h[0])  # Store plot handle for legend
            legend_text_entries.append(legendText)  # Store legend text

    # Add grid and legend to the plot
    plt.grid(True)
    plt.legend(legend, legend_text_entries, loc='upper right')
    
    # Display the plot
    plt.show()
