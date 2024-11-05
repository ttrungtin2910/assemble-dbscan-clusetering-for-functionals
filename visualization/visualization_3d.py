import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def plot_3d(data, cluster, x, y):
    """
    Plots DBSCAN clustering results in a 3D contour format with distinct colors for each cluster.

    This function visualizes clusters identified by DBSCAN by assigning unique colors to each cluster.
    Noise points are represented in black. The plot includes a legend indicating cluster labels 
    and noise, enhancing interpretability.

    Parameters
    ----------
    data : np.ndarray
        A 3D array containing data points to be plotted, with each element representing a 2D contour 
        for a cluster.
    cluster : np.ndarray
        A 1D array of cluster labels, where noise points are labeled as 0, and other integers 
        represent specific clusters.
    x : np.ndarray
        A 1D array of x-axis coordinates used for the contour plot.
    y : np.ndarray
        A 1D array of y-axis coordinates used for the contour plot.

    Returns
    -------
    None
        This function generates a plot and displays it but does not return any value.

    Notes
    -----
    - Each cluster is assigned a distinct color from an HSV colormap.
    - A legend is created based on cluster numbers, including a 'Noise' label for unclustered points.
    """
    # Plotting the clusters
    k = int(cluster.max())
    n = len(cluster)

    legend_handles=[]

    colors = cm.hsv(np.linspace(0, 1, k+1))

    plt.figure()
    for i in range(0, k + 1):
        cluster_fi = [data[j] for j in range(n) if cluster[j] == i]
        color = colors[i - 1] if i > 0 else [0, 0, 0]
        legend_text = f'Cluster #{i}' if i > 0 else 'Noise'
        
        for f_j in cluster_fi:
            plt.contour(x, y, f_j, colors=[color], linewidths=1.5)
        
        # Append a proxy artist for the legend entry (only add one per cluster)
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=legend_text))
        
    plt.title('3D Contour Plot for DBSCAN Clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    # Add legend with the handles collected
    plt.legend(handles=legend_handles)

    plt.show()
