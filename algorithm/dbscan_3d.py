import numpy as np
from typing import List

def clustering_3d(
        data: List[np.ndarray],
        epsilon: float, # Neighborhood radius (Bán kính lân cận)
        min_points: int, # Minimum number of points (Số điểm tối thiểu)
        step: float
    ) -> np.ndarray:

    """
    Performs DBSCAN clustering on a dataset using a overlap distance matrix.

    This function applies DBSCAN on a dataset `data`, with clusters identified based on 
    a neighborhood radius (`epsilon`) and a minimum number of points per cluster (`min_points`).
    The distance matrix is calculated using an integration-like similarity measure, 
    and noise points are labeled as 0, with clusters starting from label 1.

    Parameters
    ----------
    data : list[np.ndarray]
        A list of 2D array where each array is a data sample to be clustered.
    epsilon : float
        The neighborhood radius within which points are considered neighbors.
    min_points : int
        The minimum number of points required to form a dense region (cluster).
    step : float
        The incremental step size used in the overlap distance calculation.

    Returns
    -------
    np.ndarray
        A 1D array containing cluster labels for each point in the dataset, 
        with 0 indicating noise points and positive integers denoting clusters.

    Notes
    -----
    - This DBSCAN implementation uses a custom distance matrix, where each entry is calculated 
      based on an integration-like similarity measure.
    - Noise points are automatically labeled as 0.
    """

    # DBSCAN parameters

    num_clusters = 0  # Cluster count

    # Initialize DBSCAN results
    num_data = len(data)
    label = np.zeros(num_data)

    # Distance matrix calculation using integration-like similarity measure
    distance_matrix = np.zeros((num_data, num_data))
    for j in range(num_data):
        for i in range(num_data):
            diff = np.minimum(data[i], data[j])
            distance_matrix[i, j] = 1 - np.sum(diff) * step**2 + 1e-10  # Using a small constant to avoid zero distances

    # DBSCAN clustering process
    visited = np.zeros(num_data, dtype=bool)
    isnoise = np.zeros(num_data, dtype=bool)

    for i in range(num_data):
        if not visited[i]:
            visited[i] = True
            neighbors = np.where(distance_matrix[i] <= epsilon)[0]
            if len(neighbors) < min_points:
                isnoise[i] = True
            else:
                num_clusters += 1
                label[i] = num_clusters
                k = 0
                while True:
                    j = neighbors[k]
                    if not visited[j]:
                        visited[j] = True
                        neighbors_j = np.where(distance_matrix[j] <= epsilon)[0]
                        if len(neighbors_j) >= min_points:
                            # Add new neighbors (Thêm các lân cận mới)
                            for neighbor_temp in neighbors_j:
                                if neighbor_temp not in neighbors:
                                    neighbors = np.concatenate((neighbors, [neighbor_temp]))

                    if label[j] == 0:
                        label[j] = num_clusters
                    k += 1
                    if k >= len(neighbors):
                        break
                # END while loop
            # END if
        # END if
    # END for loop
    return label