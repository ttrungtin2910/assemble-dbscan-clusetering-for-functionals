import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from evaluate.metrics import ari
from algorithm.dbscan_3d import clustering_3d
from visualization.visualization_3d import plot_3d

def create_pdfs_dataset(
        x,
        y,
        visualize: bool= False,
        normalization: bool= False,
    ):
    # Means and covariances
    mu1 = [1.4, -2]
    mu2 = [-1.4, -2]
    sigma1 = np.array([[0.3, 0], [0, 0.3]])
    sigma2 = np.array([[0.1, 0], [0, 0.3]])

    # Generating data points
    data1 = np.random.multivariate_normal(mu1, sigma1, 50)
    data2 = np.random.multivariate_normal(mu2, sigma2, 50)

    if normalization:
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        data1 = scaler.fit_transform(data1)
        data2 = scaler.fit_transform(data2)

    mu = np.vstack((data1, data2))

    label_true = [0]*50 + [1]*50

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


if __name__ == "__main__":
    # Parameters
    step = 0.08
    x, y = np.meshgrid(np.arange(-6.5, 6.5 + step, step), np.arange(-6.5, 6.5 + step, step))

    f_x, f_y = create_pdfs_dataset(
        x=x,
        y=y,
        visualize=True,
        # normalization=True
    )
    
    result = {}
    list_ari = []

    range_epsilon = np.arange(0.01, 1 + step, 0.01).tolist()

    for epsilon in tqdm(range_epsilon):
        f_y_pred = clustering_3d(data=f_x,epsilon=epsilon, min_points=10,step=step)

        # plot_3d(f_x, f_y_pred, x, y)

        ari_value = ari(f_y, f_y_pred)

        # result[f'epsilon_{epsilon}'] = ari_value
        list_ari.append(ari_value)
    plt.figure()
    plt.scatter(range_epsilon, list_ari)
    # Add title and labels
    plt.title(f'Max ARI = {max(list_ari)}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Display the plot
    plt.show()
