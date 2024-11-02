import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from f_dbscan.f_dbscan import f_dbscan, plot_f_dbscan
from f_dbscan.sim_pdf_abnormal import sim_pdf_abnormal

# Set random seed for data
random_seed = 24
np.random.seed(random_seed)

def create_pdfs_dataset(data_points, visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    """
    Create dataset fx, fy by defauls

    Paramters
    ---------
    data_points: np.ndarray
    visualize : bool
        If True, visualize the output data

    Returns
    -------
    f_x : np.ndarray
        Functions create by provided data
    f_y : np.ndarray
        True label of dataset
    """
    # Generate probability density functions (PDFs) for abnormal functions
    # Generate random means for the PDFs
    mu1 = np.random.normal(10, 5, 20)
    mu2 = np.random.normal(-10, 2, 5)

    # Generate the PDFs and true labels
    f_x, f_y = sim_pdf_abnormal(
        [mu1, mu2],
        [6, 9],
        data_points
    )

    # Plot the original PDFs in gray color
    if visualize:
        plt.figure(figsize=(12, 6))
        for i in range(f_x.shape[1]):
            plt.plot(data_points, f_x[:, i], color=[0.8, 0.8, 0.8])
        plt.title('Original PDFs')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.show()
    
    return f_x, f_y 

if __name__ == '__main__':

    # Declare param input
    start = -50
    stop = 50
    step = 0.2

    # Create range data
    data_points = np.arange(start, stop + step, step)
    # # Create dataset
    f_x, f_y = create_pdfs_dataset(data_points=data_points, visualize = False)

    f_y_cluster = f_dbscan(data = f_x, epsilon = 0.5, min_points = 3, step=step)

    # Plot the DBSCAN results with annotations
    plot_f_dbscan(f_x, f_y_cluster, data_points)
