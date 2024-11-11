import numpy as np
from evaluate.metrics import ari
from algorithm.pdf_dbscan_2d import Dbscan2D
from distance.pdf_distance import Distance2D


def run():
    # Declare param input
    start = -50
    stop = 50
    step = 0.2

    # Create range data
    grid = np.arange(start, stop + step, step)

    # Create object
    object_dbscan_2d = Dbscan2D()

    # Create dataset
    data, label = object_dbscan_2d.create_dataset_random(
        grid=grid,
        visualize=False
    )
    
    # Run algorithm
    label_infer = object_dbscan_2d.run_algorithm(
        data = data,
        epsilon = 0.5,
        min_points = 3,
        step=step,
        distance = Distance2D.l1_distance,
    )

    # Evaluate
    ari_value = ari(
        labels_infer=label_infer,
        labels_true=label
    )

    print(f"ARI: {ari_value}")

    # Visulization
    object_dbscan_2d.visualize_inference(
        f=data,
        cluster=label_infer,
        grid=grid
    )
