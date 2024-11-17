from core.run_dbscan import run as run_2d
from core.run_dbscan_3d import run as run_3d
import numpy as np

# Set random seed for data
random_seed = 24
np.random.seed(random_seed)


if __name__ == '__main__':
    # run_2d()
    # run_3d(data_type=1)
    run_3d(data_type=0)