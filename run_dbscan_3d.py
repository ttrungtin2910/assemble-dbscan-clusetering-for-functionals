import numpy as np

from evaluate.metrics import ari
from algorithm.dbscan_3d import clustering_3d
from visualization.visualization_3d import plot_3d
from data_making.pdf_3d_abnormal import create_pdfs_dataset



if __name__ == "__main__":
    # Parameters
    step = 0.08
    x, y = np.meshgrid(np.arange(-6.5, 6.5 + step, step), np.arange(-6.5, 6.5 + step, step))

    f_x, f_y = create_pdfs_dataset(
        x=x,
        y=y,
        visualize=True
    )
    
    result = {}
    list_ari = []

    # for epsilon in tqdm(range_epsilon):
    f_y_pred = clustering_3d(data=f_x,epsilon=0.4, min_points=10,step=step)

    plot_3d(f_x, f_y_pred, x, y)

    ari_value = ari(f_y, f_y_pred)

    print(ari_value)
