import numpy as np

def Integration(h, fv, Dim):
    """
    Approximates the numerical integration of a function over a grid.

    Parameters
    ----------
    h : float
        The step size of the grid (uniform spacing between points).
    fv : array-like
        A NumPy array containing the function values at the grid points.
    Dim : int
        The dimensionality of the space (e.g., 1 for 1D, 2 for 2D, 3 for 3D).

    Returns
    -------
    sol : float
        The numerical approximation of the integral over the grid.

    Notes
    -----
    This function uses the Riemann sum approximation to compute the integral.
    The integral is approximated as the sum of all function values multiplied 
    by the volume (or area) of each grid element, which is given by `h^Dim`.
    
    Example
    -------
    >>> h = 0.1
    >>> fv = np.array([[1, 2], [3, 4]])
    >>> Dim = 2
    >>> Integration(h, fv, Dim)
    0.1

    """
    mesh = h ** Dim  # Volume of a single grid element
    sol = mesh * np.sum(fv)  # Riemann sum approximation
    return sol
