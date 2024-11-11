import numpy as np
from typing import Union
from sklearn.metrics import adjusted_rand_score


def ari(
        labels_true: Union[list,np.ndarray],
        labels_infer: Union[list,np.ndarray]
    ):
    """
    Caculate Adjusted Rand Index (ARI) between 2 cluster.

    Parameters
    ----------
    labels_true: np.array | List
        Ground truth.
    labels_infer: np.array | List
        Label.

    Returns
    -------
    float: Value of ARI in [-1,1]
    """
    ari_score = adjusted_rand_score(labels_true, labels_infer)
    return ari_score

