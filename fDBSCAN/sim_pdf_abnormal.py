import numpy as np
from scipy.stats import norm

def SimPDFAbnormal(
        mu_ranges,
        sig_values,
        grid,
        abnormal_params=None
    ):
    """
    Simulates Probability Density Functions (PDFs) for multiple normal distributions 
    and optional abnormal distributions.

    Parameters
    ----------
    mu_ranges : list of lists
        A list where each element is a list of mean values (`mu`) for a group of distributions.
        Each group corresponds to a different normal distribution family.
        
    sig_values : list of floats
        A list containing the standard deviations (`sigma`) for each group of distributions.
        The length of `sig_values` must match the number of groups in `mu_ranges`.

    grid : array-like
        A NumPy array representing the points over which the PDFs are evaluated.

    abnormal_params : list of lists, optional
        A list of abnormal distribution parameters, where each element is a pair of:
        - `mus`: List of means for the abnormal group.
        - `sigmas`: List of corresponding standard deviations for each mean.
        If provided, the function will generate PDFs for these abnormal distributions.

    Returns
    -------
    Data : np.ndarray
        A 2D NumPy array where each column corresponds to the PDF of a distribution,
        with rows representing values evaluated over the `grid`.

    labels : np.ndarray
        A 1D NumPy array of labels for each PDF:
        - Regular group distributions are labeled starting from 1.
        - Abnormal distributions are assigned the label `num_groups + 1`.

    Example
    -------
    >>> mu_ranges = [[0, 1], [2, 3]]
    >>> sig_values = [0.5, 1.0]
    >>> grid = np.linspace(-5, 5, 100)
    >>> abnormal_params = [[[4, 5], [0.8, 1.2]]]
    >>> Data, labels = SimPDFAbnormal(mu_ranges, sig_values, grid, abnormal_params)
    >>> print(Data.shape)
    (100, 5)  # 100 grid points, 5 PDFs (4 normal + 1 abnormal)
    >>> print(labels)
    [1 1 2 2 3]

    Notes
    -----
    - This function is useful for generating PDFs of multiple normal distributions 
      for statistical simulations or experiments.
    - The abnormal distributions are optional, and they allow for more complex scenarios 
      where the data may have outliers or separate clusters.
    """
    # Initialize variables
    num_groups = len(mu_ranges)
    pdfs = []
    labels = []

    # Generate PDFs for each group
    for group_index, mu_range in enumerate(mu_ranges):
        for mu in mu_range:
            f_single = norm.pdf(grid, loc=mu, scale=sig_values[group_index])
            pdfs.append(f_single)
            labels.append(group_index + 1)  # MATLAB indices start at 1

    # Generate PDFs for abnormal distributions
    if abnormal_params is not None:
        for abnormal_group in abnormal_params:
            if isinstance(abnormal_group, list) and len(abnormal_group) == 2:
                mus = abnormal_group[0]
                sigmas = abnormal_group[1]
                abnormal_pdf = np.zeros_like(grid)
                for mu, sigma in zip(mus, sigmas):
                    abnormal_pdf += norm.pdf(grid, loc=mu, scale=sigma)
                pdfs.append(abnormal_pdf)
                labels.append(num_groups + 1)  # Assign label for "abnormal" group

    # Convert lists to NumPy arrays
    Data = np.array(pdfs).T  # Transpose to match MATLAB's orientation
    labels = np.array(labels)

    return Data, labels
