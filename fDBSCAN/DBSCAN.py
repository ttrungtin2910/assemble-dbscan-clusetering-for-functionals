import numpy as np
import matplotlib.pyplot as plt
from fDBSCAN.plot_dbscan import PlotDBSCAN
from fDBSCAN.integration import Integration

# ===============================
# Parameters for DBSCAN
epsilon = 0.5    # Neighborhood radius (Bán kính lân cận)
MinPts = 3     # Minimum number of points (Số điểm tối thiểu)

C = 0             # Initialize number of clusters (Khởi tạo số lượng cụm)

# Initialize results for DBSCAN
n = f.shape[1]    # Number of data points
IDX = np.zeros(n, dtype=int)

# Compute the distance matrix
D = np.zeros((n, n))
for j in range(n):
    for i in range(n):
        D[i, j] = Integration(param['h'], np.abs(f[:, i] - f[:, j]), 1) + 1e-10
        # Alternative metric (commented out in MATLAB code)
        # D[i, j] = -np.log(Integration(param['h'], np.sqrt(f[:, i] * f[:, j]), 1)) + 1e-10

visited = np.zeros(n, dtype=bool)   # Track if a point has been visited (Theo dõi nếu một điểm đã được lướt qua)
isnoise = np.zeros(n, dtype=bool)   # Track if a point is classified as noise (Theo dõi nếu một điểm được phân loại là nhiễu)

# DBSCAN clustering process (Quá trình phân cụm DBSCAN)
for i in range(n):
    if not visited[i]:
        visited[i] = True

        # Find neighbors of point i within epsilon distance (Tìm các lân cận của điểm i trong khoảng cách epsilon)
        Neighbors = np.where(D[i, :] <= epsilon)[0]
        if len(Neighbors) < MinPts:   # If not enough neighbors, classify as noise (Nếu không đủ lân cận, phân loại là nhiễu)
            isnoise[i] = True
        else:
            C += 1
            IDX[i] = C                # Assign cluster label to point i (Gán nhãn cụm cho điểm i)

            # Expand cluster (Mở rộng cụm)
            k = 0
            while True:
                j = Neighbors[k]
                print(k)
                print(j)
                print('---')

                if not visited[j]:
                    visited[j] = True
                    Neighbors2 = np.where(D[j, :] <= epsilon)[0]  # Find neighbors (Tìm lân cận)
                    if len(Neighbors2) >= MinPts:
                        # Add new neighbors (Thêm các lân cận mới)
                        

                        for neighbor_temp in Neighbors2:
                            if neighbor_temp not in Neighbors:
                                Neighbors = np.concatenate((Neighbors, [neighbor_temp]))

                if IDX[j] == 0:
                    # Assign current cluster ID to point j (Gán ID cụm hiện tại cho điểm j)
                    IDX[j] = C
                # PlotDBSCAN(f, IDX, param)
                # Move to the next neighbor (Chuyển đến lân cận tiếp theo)
                k += 1
                
                if k >= len(Neighbors):
                    break


# Plot the DBSCAN results with annotations (Vẽ kết quả DBSCAN với chú thích)
PlotDBSCAN(f, IDX, param)

pass