import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def create_pdfs_dataset(
    x: np.ndarray,
    y: np.ndarray,
    visualize: bool = False,
):
    """
    Create a dataset of probability density functions (PDFs) based on generated data points.

    Parameters
    ----------
    x : ndarray
        A 2D array representing the grid of x-coordinates for evaluating the PDFs.
    y : ndarray
        A 2D array representing the grid of y-coordinates for evaluating the PDFs.
    visualize : bool, optional
        If set to True, plots the contour of the generated PDFs (default is False).

    Returns
    -------
    tuple
        A tuple containing:
        - fi (list of ndarray): A list of PDF values evaluated over the grid defined by x and y.
        - label_true (list of int): A list of true labels corresponding to each data point.

    Notes
    -----
    This function generates three clusters of data points using multivariate normal distributions 
    with predefined means and covariances. It then computes the PDF for each data point over the 
    provided grid using randomly generated scaling factors. If visualization is enabled, the 
    function plots the contour lines of the PDFs.

    Example
    -------
    ```python
    x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    fi, labels = create_pdfs_dataset(x, y, visualize=True)
    ```
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
        fi_j = rv.pdf(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
        # Check if normalization
        fi.append(fi_j)
    # Plotting the contours
    if visualize:
        plt.figure()
        for fi_j in fi:
            plt.contour(x, y, fi_j)
        plt.title('Initial Data Points Contours')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    return fi, label_true