import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def PlotDBSCAN(f, IDX, param):
    """
    Plots the results of DBSCAN clustering with clusters and noise points 
    displayed using different colors.

    Parameters
    ----------
    f : np.ndarray
        A 2D NumPy array where each column represents a data point's values over 
        a certain range (e.g., function values along the x-axis).
    
    IDX : np.ndarray
        A 1D array of cluster labels assigned by DBSCAN, where `0` represents noise points 
        and positive integers represent different clusters.
    
    param : dict
        A dictionary containing plot parameters. It must have the key:
        - 'x': A 1D array representing the x-axis values for plotting.

    Notes
    -----
    - Noise points are displayed in black.
    - Each cluster is assigned a unique color using the HSV colormap.
    - The plot includes a legend showing each cluster and noise.

    Example
    -------
    >>> f = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    >>> IDX = np.array([1, 1, 0])
    >>> param = {'x': np.array([0, 1, 2])}
    >>> PlotDBSCAN(f, IDX, param)
    
    This will plot two data series: one for cluster #1 and one for noise.

    """
    # Determine the number of clusters (excluding noise labeled as 0)
    k = np.max(IDX)

    # Generate colors for each cluster using HSV colormap
    Colors = cm.hsv(np.linspace(0, 1, k+1))

    # Prepare lists for legend handles and text entries
    Legends = []
    legendTextEntries = []

    # Iterate over clusters (including noise with index 0)
    for i in range(k + 1):
        # Select data points belonging to the current cluster (or noise)
        fi = f[:, IDX == i]

        if i != 0:  # If not noise
            Color = Colors[i - 1]  # Assign color to the cluster
            legendText = f'Cluster #{i}'
        else:  # Noise points
            Color = [0, 0, 0]  # Black color for noise
            legendText = 'Noise'

        # Plot only if there are data points in the current group
        if fi.size > 0:
            h = plt.plot(param['x'], fi, color=Color)  # Plot data points
            Legends.append(h[0])  # Store plot handle for legend
            legendTextEntries.append(legendText)  # Store legend text

    # Add grid and legend to the plot
    plt.grid(True)
    plt.legend(Legends, legendTextEntries, loc='upper right')
    
    # Display the plot
    plt.show()
