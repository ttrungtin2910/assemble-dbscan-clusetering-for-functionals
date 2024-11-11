import numpy as np
from evaluate.metrics import ari
from algorithm.pdf_dbscan import Dbscan_3D
from distance.pdf_distance import Distance3D

def run():
    # Declare param input
    step = 0.08

    # Create grid data
    grid_x, grid_y = np.meshgrid(
        np.arange(-6.5, 6.5 + step, step),
        np.arange(-6.5, 6.5 + step, step)
    )

    # Create object
    object_dbscan_3d = Dbscan_3D()

    # Create dataset
    data, label = object_dbscan_3d.create_dataset_random(
        grid_x=grid_x,
        grid_y=grid_y,
        visualize=False
    )

    # Run algorithm
    label_infer = object_dbscan_3d.run_algorithm(
        data = data,
        epsilon = 0.5,
        min_points = 3,
        step = step,
        distance = Distance3D.overlap_distance,
    )

    # Evaluate
    ari_value = ari(
        labels_infer=label_infer,
        labels_true=label
    )

    print(f"ARI: {ari_value}")

    # Visulization
    object_dbscan_3d.visualize_inference(
        f=data,
        cluster=label_infer,
        grid_x=grid_x,
        grid_y=grid_y
    )