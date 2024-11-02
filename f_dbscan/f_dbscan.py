import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def l1_distance(
        data1: np.ndarray,
        data2: np.ndarray,
        data_step: float,
        num_dim: int
    ) -> float:
    """
    Calculates the L1 (Manhattan) distance between two data points or arrays on a discretized grid.

    This function computes the L1 distance by taking the element-wise absolute difference 
    between two input data arrays (`data1` and `data2`). It then uses a Riemann sum approximation
    based on the specified grid spacing (`data_step`) and dimensionality (`num_dim`) to calculate 
    the distance.

    Parameters
    ----------
    data1 : np.ndarray
        The first data point or array representing a set of values.
        
    data2 : np.ndarray
        The second data point or array to compare with `data1`.
        
    data_step : float
        The spacing between grid points, used to scale the distance measurement.
        
    num_dim : int
        The number of dimensions in the grid, determining the scale of the mesh volume.

    Returns
    -------
    float
        The L1 distance between `data1` and `data2`, adjusted by the mesh volume and a small constant 
        (1e-10) added for numerical stability.

    Example
    -------
    >>> data1 = np.array([1, 2, 3])
    >>> data2 = np.array([4, 5, 6])
    >>> data_step = 0.1
    >>> num_dim = 3
    >>> distance = l1_distance(data1, data2, data_step, num_dim)
    >>> print(distance)
    0.009001  # Example output (actual result depends on inputs)

    Notes
    -----
    - This function is suitable for cases where the L1 distance needs to be estimated on a discretized 
    space with specific grid spacing.
    - The small constant added to the result (1e-10) helps prevent issues with exact zero in numerical computations.
    """
    fv = np.abs(data1 - data2)
    mesh = data_step ** num_dim  # Volume of a single grid element
    sol = mesh * np.sum(fv)  # Riemann sum approximation
    return sol + 1e-10

def f_dbscan(
        data,
        epsilon: float, # Neighborhood radius (Bán kính lân cận)
        min_points: int, # Minimum number of points (Số điểm tối thiểu)
        step: float
    ) -> np.ndarray:
    """
    Applies the DBSCAN algorithm 
    to cluster data points based on density and distance criteria.

    The function performs DBSCAN clustering by calculating an L1 distance matrix, 
    using each data point's neighbors within a specified radius (`epsilon`) and 
    a minimum point threshold (`min_points`). Points that don't meet the density 
    requirements are classified as noise.

    Parameters
    ----------
    data : np.ndarray
        A 2D array representing data points.
        
    epsilon : float
        The radius of the neighborhood for each point, used to define dense regions 
        where clusters can form.
        
    min_points : int
        The minimum number of points required in a neighborhood to consider a point 
        part of a cluster.
        
    step : float
        The step size used for calculating the distance metric.

    Returns
    -------
    np.ndarray
        A 1D array of integer labels where each element corresponds to a data point's 
        cluster assignment. Noise points are labeled as 0, while other points have 
        positive integer labels indicating their cluster membership.

    Notes
    -----
    - This implementation uses the L1 distance metric and a Riemann sum approximation for distance.
    - `fDBSCAN` can be applied to high-dimensional data but may require tuning `epsilon` 
        and `min_points` based on the density of the data.
    - Noise points are labeled as `0` and do not belong to any cluster.
    """
    # data = np.array([norm.pdf(grid, loc=5, scale=1) if i >= 5 else norm.pdf(grid, loc=0, scale=1) for i in range(10)]).T


    num_clusters = 0
    # Number of data points
    num_data = data.shape[1]

    # Initialize results for DBSCAN
    label = np.zeros(num_data, dtype=int)

    # Compute the distance matrix
    distance_matrix = np.zeros((num_data, num_data))

    # Calculate distance matrix
    for j in range(num_data):
        for i in range(num_data):
            # Calculate distance
            distance_matrix[i, j] = l1_distance(
                data1=data[:, i],
                data2=data[:, j],
                data_step=step,
                num_dim=1
            ) 
            # Alternative metric (commented out in MATLAB code)
            # D[i, j] = -np.log(Integration(param['h'], np.sqrt(f[:, i] * f[:, j]), 1)) + 1e-10

    # Create temp variaables
    visited = np.zeros(num_data, dtype=bool)   # Track if a point has been visited (Theo dõi nếu một điểm đã được lướt qua)
    isnoise = np.zeros(num_data, dtype=bool)   # Track if a point is classified as noise (Theo dõi nếu một điểm được phân loại là nhiễu)

    # DBSCAN clustering process (Quá trình phân cụm DBSCAN)
    for i in range(num_data):
        if not visited[i]:
            visited[i] = True

            # Find neighbors of point i within epsilon distance (Tìm các lân cận của điểm i trong khoảng cách epsilon)
            neighbors = np.where(distance_matrix[i, :] <= epsilon)[0]

            if len(neighbors) < min_points:   # If not enough neighbors, classify as noise (Nếu không đủ lân cận, phân loại là nhiễu)
                isnoise[i] = True
            else:
                num_clusters += 1
                label[i] = num_clusters                # Assign cluster label to point i (Gán nhãn cụm cho điểm i)

                # Expand cluster (Mở rộng cụm)
                k = 0
                while True:
                    j = neighbors[k]
                    
                    # Check if point data not belong to any cluster
                    if not visited[j]:
                        visited[j] = True

                        # HGet list neighbor
                        neighbors_j = np.where(distance_matrix[j, :] <= epsilon)[0]  # Find neighbors (Tìm lân cận)
                        if len(neighbors_j) >= min_points:

                            # Add new neighbors (Thêm các lân cận mới)
                            for neighbor_temp in neighbors_j:
                                if neighbor_temp not in neighbors:
                                    neighbors = np.concatenate((neighbors, [neighbor_temp]))

                    if label[j] == 0:
                        # Assign current cluster ID to point j (Gán ID cụm hiện tại cho điểm j)
                        label[j] = num_clusters
                    # PlotDBSCAN(f, IDX, param)
                    # Move to the next neighbor (Chuyển đến lân cận tiếp theo)
                    k += 1
                    
                    if k >= len(neighbors):
                        break
                # END loop while
            # END if else
        # END if else
    # END loop for
    return label


def plot_f_dbscan(
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