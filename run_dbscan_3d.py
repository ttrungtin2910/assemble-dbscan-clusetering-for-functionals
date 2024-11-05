import numpy as np
import matplotlib.pyplot as plt
from algorithm.dbscan_3d import clustering_3d
from visualization.visualization_3d import plot_3d
from scipy.stats import multivariate_normal

def create_pdfs_dataset(x, y, visualize: bool= False):
    # Means and covariances
    mu1 = [4, -2]
    mu2 = [-4, -2]
    sigma1 = np.array([[0.3, 0], [0, 0.3]])
    sigma2 = np.array([[0.1, 0], [0, 0.3]])

    # Generating data points
    data1 = np.random.multivariate_normal(mu1, sigma1, 50)
    data2 = np.random.multivariate_normal(mu2, sigma2, 50)
    mu = np.vstack((data1, data2))

    # Initialize variables for contour plotting
    fi = []
    sig = []
    num_sample = mu.shape[0]
    if visualize:
        plt.figure()
        

    for j in range(num_sample):
        sig_j = 1 # np.random.rand()#3 + 5 * np.random.rand()
        rv = multivariate_normal(mean=mu[j], cov=np.eye(2) / sig_j)
        fi_j = rv.pdf(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
        if visualize:
            plt.contour(x, y, fi_j)
        fi.append(fi_j)
    # Plotting the contours
    if visualize:
        plt.title('Initial Data Points Contours')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    return fi


if __name__ == "__main__":
    # Parameters
    step = 0.08
    x, y = np.meshgrid(np.arange(-6.5, 6.5 + step, step), np.arange(-6.5, 6.5 + step, step))

    f_x = create_pdfs_dataset(x=x, y=y, visualize=True)
    
    f_y = clustering_3d(data=f_x,epsilon=0.5, min_points=10,step=step)

    plot_3d(f_x, f_y, x, y)
