import numpy as np

class Distance2D:
    '''
    This class list all measurements to calcuate distance between 2 pdf

    Static methods
    --------------
    - l1_distance
    
    '''

    @staticmethod
    def l1_distance(
        data1: np.ndarray,
        data2: np.ndarray,
        data_step: float,
        num_dim: int
    ) -> float:
        """
        Calculates the L1 (Manhattan) distance between two data points
        or arrays on a discretized grid.

        This function computes the L1 distance by taking the element-wise
        absolute difference between  input data arrays (`data1` and `data2`).
        It then uses a sum approximation based on the specified grid spacing
        (`data_step`) and dimensionality (`num_dim`) to calculate distance.

        Parameters
        ----------
        data1 : np.ndarray
            The first data point or array representing a set of values.
            
        data2 : np.ndarray
            The second data point or array to compare with `data1`.
            
        data_step : float
            The spacing between grid points, used to scale the distance
            measurement.
            
        num_dim : int
            The number of dimensions in the grid, determining the scale
            of the mesh volume.

        Returns
        -------
        float
            The L1 distance
        """
        fv = np.abs(data1 - data2)
        # Volume of a single grid element
        mesh = data_step ** num_dim
        # Sum approximation
        sol = mesh * np.sum(fv)
        return sol + 1e-10
    

class Distance3D:
    '''
    This class list all measurements to calcuate distance between 2 pdf

    Static methods
    --------------
    - overlap_distance
    
    '''
    @staticmethod
    def overlap_distance(
        data1: np.ndarray,
        data2: np.ndarray,
        data_step: float
    ):
        diff = np.minimum(data1, data2)
        return 1 - np.sum(diff) * data_step**2 + 1e-10  