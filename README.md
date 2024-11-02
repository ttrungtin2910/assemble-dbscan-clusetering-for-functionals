# Clustering Probability Density Functions using DBSCAN

This repository provides a Python implementation of clustering generated Probability Density Functions (PDFs) using the DBSCAN algorithm. The code simulates PDFs with specified means and standard deviations, performs clustering, and visualizes the results.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Understanding the Output](#understanding-the-output)
- [Code Explanation](#code-explanation)
  - [Simulating PDFs](#simulating-pdfs)
  - [DBSCAN Clustering](#dbscan-clustering)
  - [Visualization](#visualization)
- [Customization](#customization)
- [License](#license)

## Introduction

This project demonstrates how to:

- Generate multiple sets of PDFs with specified means and standard deviations.
- Compute a distance matrix based on the difference between PDFs.
- Cluster the PDFs using the DBSCAN algorithm.
- Visualize the clustering results with different colors for each cluster and noise points.

## Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- NumPy
- SciPy
- Matplotlib

You can install the required Python packages using:

```bash
pip install numpy scipy matplotlib
```

## Installation

1. Clone the repository or download the script:

   ```bash
   git clone ...
   ```

2. Navigate to the project directory:

   ```bash
   cd dbscan_pdf_clustering
   ```

3. Ensure all dependencies are installed (see [Prerequisites](#prerequisites)).

## Usage

### Running the Script

Run the Python script using:

```bash
python dbscan_clustering.py
```

This will execute the script and display the clustering results in a plot.

### Understanding the Output

- **Original PDFs Plot**: Displays all generated PDFs in gray color before clustering.
- **DBSCAN Clustering Results**: Shows the clustered PDFs with each cluster in a different color and noise points in black.

## Code Explanation

The script consists of the following main parts:

### Simulating PDFs

```python
# Define simulation parameters
param = {}
param['h'] = 0.2
param['x'] = np.arange(-50, 50 + param['h'], param['h'])

# Generate random means for the PDFs
mu1 = np.random.normal(10, 5, 20)
mu2 = np.random.normal(-10, 2, 5)

# Generate the PDFs and true labels
f, param['truelabels'] = SimPDFAbnormal(
    [mu1, mu2],
    [6, 9],
    param['x']
)
```

- **`param['h']`**: Spacing between points in the grid.
- **`param['x']`**: X-axis values for evaluating the PDFs.
- **`mu1` and `mu2`**: Arrays of means for two groups of PDFs.
- **`SimPDFAbnormal`**: Function that generates the PDFs based on provided means and standard deviations.

### DBSCAN Clustering

```python
# Parameters for DBSCAN
epsilon = 0.5     # Neighborhood radius
MinPts = 3        # Minimum number of points

C = 0             # Initialize number of clusters

# Initialize results for DBSCAN
n = f.shape[1]    # Number of data points
IDX = np.zeros(n, dtype=int)

# Compute the distance matrix
D = np.zeros((n, n))
for j in range(n):
    for i in range(n):
        D[i, j] = Integration(param['h'], np.abs(f[:, i] - f[:, j]), 1) + 1e-10

visited = np.zeros(n, dtype=bool)   # Track if a point has been visited
isnoise = np.zeros(n, dtype=bool)   # Track if a point is classified as noise

# DBSCAN clustering process
for i in range(n):
    if not visited[i]:
        visited[i] = True

        # Find neighbors of point i within epsilon distance
        Neighbors = np.where(D[i, :] <= epsilon)[0]
        if len(Neighbors) < MinPts:
            isnoise[i] = True
        else:
            C += 1
            IDX[i] = C  # Assign cluster label to point i

            # Expand cluster
            k = 0
            while True:
                j = Neighbors[k]

                if not visited[j]:
                    visited[j] = True
                    Neighbors2 = np.where(D[j, :] <= epsilon)[0]
                    if len(Neighbors2) >= MinPts:
                        # Add new neighbors
                        Neighbors = np.concatenate((Neighbors, Neighbors2))
                        Neighbors = np.unique(Neighbors)

                if IDX[j] == 0:
                    # Assign current cluster ID to point j
                    IDX[j] = C

                # Move to the next neighbor
                k += 1
                if k >= len(Neighbors):
                    break
```

- **`epsilon`**: Maximum distance to consider two points as neighbors.
- **`MinPts`**: Minimum number of neighbors to form a dense region.
- **`D`**: Distance matrix computed using numerical integration of the absolute difference between PDFs.
- The algorithm iterates over each point, expanding clusters where density conditions are met.

### Visualization

```python
# Plot the DBSCAN results with annotations
PlotDBSCAN(f, IDX, param)
```

- **`PlotDBSCAN`**: Function that visualizes the clustering results.
- Each cluster is plotted in a unique color, and noise points are plotted in black.
- The plot includes a legend indicating cluster numbers and noise.

## Customization

You can customize the script by adjusting the following:

- **Simulation Parameters**:
  - Change `param['h']` and `param['x']` to adjust the grid spacing and range.
  - Modify the means (`mu1`, `mu2`) and standard deviations in `SimPDFAbnormal` to generate different PDFs.

- **DBSCAN Parameters**:
  - Adjust `epsilon` and `MinPts` to change the sensitivity of the clustering algorithm.

- **Distance Metric**:
  - Replace the distance computation in the `D` matrix with an alternative metric if desired. An alternative (commented out) is provided in the code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: The code uses random number generation for simulating PDFs. Results will vary with each run unless a random seed is set.

**Example**:

To set a random seed for reproducibility, add the following line before generating random means:

```python
np.random.seed(42)
```

This will ensure the same random numbers are generated each time.

---