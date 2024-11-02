import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def f_dbscan_3d(
        data,
        epsilon: float, # Neighborhood radius (Bán kính lân cận)
        min_points: int, # Minimum number of points (Số điểm tối thiểu)
        step: float
    ) -> np.ndarray:

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
                while k < len(neighbors):
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


def plot_f_dbscan_3d(data, cluster, x, y):
    # Plotting the clusters
    k = int(cluster.max())
    n = len(cluster)

    Colors = plt.cm.hsv(np.linspace(0, 1, k))

    plt.figure()
    for i in range(0, k + 1):
        cluster_fi = [data[j] for j in range(n) if cluster[j] == i]
        Color = Colors[i - 1] if i > 0 else [0, 0, 0]
        legendText = f'Cluster #{i}' if i > 0 else 'Noise'
        
        for f_j in cluster_fi:
            plt.contour(x, y, f_j, colors=[Color], linewidths=1.5)
        
    plt.title('3D Contour Plot for DBSCAN Clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()
