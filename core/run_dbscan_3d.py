import numpy as np
from evaluate.metrics import ari
from algorithm.pdf_dbscan_3d import Dbscan_3D
from distance.pdf_distance import Distance3D
from tools.log import logger

def run(data_type: int = 0):
    '''
    Parameters
    ----------
    data_type: int
        - `0`: random data
        - `1`: circle data
        - `2`: moon data
    
    '''
    # Declare param input
    logger.info('Initializing object')
    step = 0.08

    # Create object
    object_dbscan_3d = Dbscan_3D()
    logger.info('-'*50)

    # Select method to create datapoint
    logger.info('START Creating data')


    point_data, data, label, grid_x, grid_y = object_dbscan_3d.create_dataset(
        n_samples=500,
        step = step,
        index_datatype=data_type,
        visualize=True
    )
    

    logger.info('-'*50)


    logger.info('START Running algorithm')
    # Run algorithm
    label_infer = object_dbscan_3d.run_algorithm(
        data = data,
        epsilon = 0.5,
        min_points = 3,
        step = step,
        distance = Distance3D.overlap_distance,
    )
    logger.info('-'*50)

    logger.info('START Evaluate algorithm')
    # Evaluate
    ari_value = ari(
        labels_infer=label_infer,
        labels_true=label
    )
    
    logger.info(f"ARI: {ari_value}")
    logger.info('-'*50)

    logger.info('START Visualization clustering result')
    # Visulization
    object_dbscan_3d.visualize_inference(
        f=data,
        cluster=label_infer,
        grid_x=grid_x,
        grid_y=grid_y,
        point_data = point_data
    )
    logger.info('-'*50 + '\n'*2)