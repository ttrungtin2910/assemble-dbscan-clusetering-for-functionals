
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.log import logger
from evaluate.metrics import ari
from algorithm.pdf_dbscan_3d import Dbscan_3D
from distance.pdf_distance import Distance3D


def run(
        data_type: int = 0,
        random_seed: int = 24
    ):
    '''
    Parameters
    ----------
    ```self.dict_data_type = {
            0: 'random_data',
            1: 'noisy_circles',
            2: 'noisy_moons',
            3: 'varied',
            4: 'aniso',
            5: 'blobs',
            6: 'no_structure'
        }```
    
    '''
    # Declare param input
    logger.info('Initializing object')
    step = 0.08

    # Create object
    object_dbscan_3d = Dbscan_3D()

    data_points_type = object_dbscan_3d.dict_data_type.get(data_type, 'random_data')

    logger.info('-'*50)

    # Select method to create datapoint
    logger.info('START Creating data')

    num_samples = 500

    point_data, data, label, grid_x, grid_y = object_dbscan_3d.create_dataset(
        n_samples=num_samples,
        step = step,
        data_points_type=data_points_type,
        visualize=True,
        random_seed=random_seed
    )

    logger.info('-'*50)

    logger.info('START Running algorithm')
    # Create array min point range
    min_point_range = np.arange(0.25 / 100, 2 / 100 + 0.00001, 0.25 / 100)*num_samples

    # Round array
    min_point_range = np.ceil(min_point_range)

    # Min value in array is 1
    min_point_range[min_point_range < 1] = 1

    # Create epsilon range
    epsilon_range = np.arange(0.2, 0.6 + 0.00001, 0.1)

    # Create list input parameter
    list_input_parameter = [
            {"epsilon": epsilon, "min_points": min_point}
                for epsilon in epsilon_range
                    for min_point in min_point_range
            ]

    # Create figure and subplots
    num_plots = len(list_input_parameter)
    num_cols = 3  # Number of columns in the subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the required number of rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten 2D array of axes to 1D for easy access

    list_label_infer = []

    # Run the algorithm and plot results in each subplot
    for idx, input_parameter in enumerate(tqdm(list_input_parameter)):
        # Run the DBSCAN algorithm
        label_infer = object_dbscan_3d.run_algorithm(
            data=data,
            step=step,
            distance=Distance3D.overlap_distance,
            **input_parameter
        )

        list_label_infer.append(label_infer)

        # Evaluate the clustering result using ARI
        ari_value = ari(
            labels_infer=label_infer,
            labels_true=label
        )
        evaluate_status = f"ARI: {round(ari_value, 3)} when {input_parameter}"
        # logger.info(evaluate_status)
        # logger.info('-' * 50)

        # Plot the result in the corresponding subplot
        ax = axes[idx]
        object_dbscan_3d.visualize_inference(
            f=data,
            cluster=label_infer,
            grid_x=grid_x,
            grid_y=grid_y,
            point_data=point_data,
            name=f"{data_points_type}_{random_seed}",
            description=evaluate_status,
            ax=ax  # Pass the current subplot
        )

    # Hide unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    # Save the combined plot
    output_filename = os.path.join(
        object_dbscan_3d.outdir,
        f"{object_dbscan_3d.current_time}_combined_visualization.png"
    )
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


    logger.info('START Visualization clustering result as graph')
    
    object_dbscan_3d.visualize_result_as_graph(
        point_data=point_data,
        list_label_infer=list_label_infer
    )

    logger.info('-'*50 + '\n'*2)