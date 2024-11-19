import numpy as np
from scipy.special import comb

import numpy as np

import numpy as np

def Compute_distance(f1, f2, h, Dim, NormP):
    # Determine the mesh based on the dimension
    if Dim == '1D':
        mesh = h
    elif Dim == '2D':
        mesh = h * h
    elif Dim == '3D':
        mesh = h * h * h
    else:
        raise ValueError("Unsupported dimension specified. Please use '1D', '2D', or '3D'.")

    # Compute the distance based on the specified norm
    if NormP == 'L1':
        sol = mesh * np.sum(np.abs(f1 - f2))  # L1 norm
    elif NormP == 'L2':
        sol = np.sqrt(mesh * np.sum((f1 - f2) ** 2))  # L2 norm
    else:
        raise ValueError("Unsupported norm specified. Please use 'L1' or 'L2'.")

    return sol


def Fitness_func_given_ClusterResult(wf_old, ofv, h, Dim, Object_cluster, NumCluster, Criterion):
    SizeData = ofv.shape
    NumGrid = SizeData[0]  # Number of grid
    NumFunc = SizeData[1]  # Number of functions

    # Produce the cluster centroid
    centroid = np.zeros((NumGrid, NumCluster))
    ClusterNumData = np.zeros(NumCluster)
    ClusterDisDataCent = np.zeros(NumCluster)

    for i in range(NumCluster):
        ClusterNumData[i] = 0
        coordinate = np.zeros(NumGrid)
        for j in range(NumFunc):
            if Object_cluster[j] == i + 1:  # Adjust for 1-based indexing in MATLAB
                coordinate += ofv[:, j]
                ClusterNumData[i] += 1
        if ClusterNumData[i] > 0:
            centroid[:, i] = coordinate / ClusterNumData[i]

    for i in range(NumFunc):
        Idx = Object_cluster[i] - 1  # Adjust for 1-based indexing
        SE = Compute_distance(ofv[:, i], centroid[:, Idx], h, Dim, 'L2') ** 2  # Assume Compute_distance is defined
        ClusterDisDataCent[Idx] += SE

    # Sum of squared error (SSE)
    SSE = np.sum(ClusterDisDataCent)

    # Variance ratio criterion (VRC)
    mean = np.sum(ofv, axis=1) / NumFunc
    Inter = 0
    for i in range(NumCluster):
        if ClusterNumData[i] != 0:
            Inter += ClusterNumData[i] * (Compute_distance(centroid[:, i], mean, h, Dim, 'L2') ** 2)

    VRC = (Inter / SSE) * (NumFunc - NumCluster) / (NumCluster - 1) if NumCluster > 1 else 0  # Handle division by zero

    # Return fitness based on the specified criterion
    if Criterion == 'SSE':
        fitness = SSE
    elif Criterion == 'VRC':
        fitness = VRC
    else:
        raise ValueError("Unsupported criterion specified. Please use 'SSE' or 'VRC'.")

    return fitness


def Integration(h, fv, Dim):
    if Dim == '1D':
        mesh = h
    elif Dim == '2D':
        mesh = h * h
    elif Dim == '3D':
        mesh = h * h * h
    else:
        raise ValueError("Unsupported dimension specified. Please use '1D', '2D', or '3D'.")

    sol = mesh * sum(fv)
    return sol

import numpy as np

def SU_cluster_results(d, n):
    # Initialize the class matrix
    class_matrix = np.zeros((n, n), dtype=int)
    for ii in range(n):
        class_matrix[ii, 0] = ii + 1  # MATLAB is 1-based index

    # Calculate the average distance
    r = np.sum(d) / (n * n)

    # Main clustering logic
    for ii in range(n):
        for jj in range(ii + 1, n):
            if d[ii, jj] / r < 1.0e-4:
                rr = ii
                is_save = 'N'
                
                # Check whether has been saved, and record its row index
                for kk in range(n):
                    for zz in range(n):
                        if class_matrix[kk, zz] == ii + 1:
                            rr = kk
                            is_save = 'Y'
                            break
                    if is_save == 'Y':
                        break

                for cc in range(n):
                    if class_matrix[rr, cc] == jj + 1:
                        break
                    if class_matrix[rr, cc] == 0:
                        class_matrix[rr, cc] = jj + 1
                        class_matrix[jj, :] = 0
                        break

    # Shows the convergence results (Number)
    NumCluster = 0
    for ii in range(n):
        if class_matrix[ii, 0] == 0:
            continue
        NumCluster += 1
        class_set = '{'
        for jj in range(n):
            if class_matrix[ii, jj] == 0:
                break
            class_set += str(class_matrix[ii, jj])
            if jj < n - 1 and class_matrix[ii, jj + 1] != 0:
                class_set += ','
        class_set += '}'
        # You can uncomment this line to print results if needed
        # print(f'*. Class {NumCluster} : {class_set}')

    # Assign the objects to the corresponding cluster
    Object_cluster = np.zeros(n, dtype=int)
    NumCluster = 0
    for ii in range(n):
        if class_matrix[ii, 0] == 0:
            continue
        NumCluster += 1
        for jj in range(n):
            if class_matrix[ii, jj] == 0:
                break
            Object_cluster[class_matrix[ii, jj] - 1] = NumCluster

    return Object_cluster, NumCluster


import numpy as np

def SU_cluster_results(d, n):
    # Initialize the class matrix
    class_matrix = np.zeros((n, n), dtype=int)
    for ii in range(n):
        class_matrix[ii, 0] = ii + 1  # MATLAB is 1-based index

    # Calculate the average distance
    r = np.sum(d) / (n * n)

    # Main clustering logic
    for ii in range(n):
        for jj in range(ii + 1, n):
            if d[ii, jj] / r < 1.0e-4:
                rr = ii
                is_save = 'N'
                
                # Check whether has been saved, and record its row index
                for kk in range(n):
                    for zz in range(n):
                        if class_matrix[kk, zz] == ii + 1:
                            rr = kk
                            is_save = 'Y'
                            break
                    if is_save == 'Y':
                        break

                for cc in range(n):
                    if class_matrix[rr, cc] == jj + 1:
                        break
                    if class_matrix[rr, cc] == 0:
                        class_matrix[rr, cc] = jj + 1
                        class_matrix[jj, :] = 0
                        break

    # Shows the convergence results (Number)
    NumCluster = 0
    for ii in range(n):
        if class_matrix[ii, 0] == 0:
            continue
        NumCluster += 1
        class_set = '{'
        for jj in range(n):
            if class_matrix[ii, jj] == 0:
                break
            class_set += str(class_matrix[ii, jj])
            if jj < n - 1 and class_matrix[ii, jj + 1] != 0:
                class_set += ','
        class_set += '}'
        # You can uncomment this line to print results if needed
        # print(f'*. Class {NumCluster} : {class_set}')

    # Assign the objects to the corresponding cluster
    Object_cluster = np.zeros(n, dtype=int)
    NumCluster = 0
    for ii in range(n):
        if class_matrix[ii, 0] == 0:
            continue
        NumCluster += 1
        for jj in range(n):
            if class_matrix[ii, jj] == 0:
                break
            Object_cluster[class_matrix[ii, jj] - 1] = NumCluster

    return Object_cluster, NumCluster


def method_GammaSUP(Para_s, Para_tau, ofv, h, Dim, n, ng):
    MaxIter = 500
    Epsilon = 1e-6
    
    wf_old = np.eye(n)  # The weight of each function
    K_new = np.zeros((n, n))
    
    for t in range(MaxIter + 1):
        # Update pairwise distance
        d = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Construct function value of abs(f(i) - f(j))
                fvi = np.zeros(ng)
                fvj = np.zeros(ng)
                for k in range(n):
                    fvi += wf_old[i, k] * ofv[:, k]
                    fvj += wf_old[j, k] * ofv[:, k]
                d[i, j] = Integration(h, np.abs(fvi - fvj), Dim)  # Integration is assumed to be defined elsewhere

        for i in range(n):
            d[i, i] = 0.0
        for i in range(1, n):
            for j in range(i):
                d[i, j] = d[j, i]

        # Averaged pairwise distance
        r = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                r += d[i, j]
        r /= comb(n, 2)

        # Update weight
        wf_new = np.zeros((n, n))
        for i in range(n):
            sum_K = 0
            for j in range(n):
                tmp_u = -((d[i, j] / r) ** 2) / (Para_tau ** 2)
                K_new[i, j] = max(1 + Para_s * tmp_u, 0) ** (1 / Para_s)
                sum_K += K_new[i, j]

            for j in range(n):
                wf_new[i, :] += (K_new[i, j] / sum_K) * wf_old[j, :]

        # Check convergence
        if np.sum(np.abs(wf_new - wf_old)) < Epsilon:
            break
        else:
            wf_old = wf_new

    # Return cluster results
    Object_cluster, NumCluster = SU_cluster_results(d, n)  # SU_cluster_results is assumed to be defined elsewhere
    Fitness = Fitness_func_given_ClusterResult(wf_old, ofv, h, Dim, Object_cluster, NumCluster, 'VRC')  # Fitness_func_given_ClusterResult is assumed to be defined elsewhere

    return Object_cluster, Fitness
