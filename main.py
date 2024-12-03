# from core.run_dbscan import run as run_2d
from core.run_dbscan_3d import run as run_3d
import numpy as np


if __name__ == '__main__':

    random_seed = 3
    
    # run_2d()
    # for random_seed in range(0, 20):
    np.random.seed(random_seed)
    for data_type in range(6):
        # data_type = 0
        run_3d(data_type=data_type, random_seed = random_seed)
    # run_3d(data_type=1, random_seed = random_seed)