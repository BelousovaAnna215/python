import numpy as np
import typing
from collections import defaultdict
    
def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val 
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    all_ind = np.arange(num_objects)
    step = num_objects // num_folds
    res_ind = []
    for fold in range(num_folds - 1):
        left_ind = np.concatenate((all_ind[:step * fold], all_ind[step * (fold + 1):]), dtype=np.int32, casting='unsafe')
        right_ind = all_ind[step * fold:step * (fold + 1)]
        res_ind.append((left_ind, right_ind))
    left_ind = all_ind[:step * (num_folds - 1)]
    right_ind = all_ind[step * (num_folds - 1):]
    res_ind.append((left_ind, right_ind))
    return res_ind

def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations) 

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    result = dict()
    for normalizers in parameters['normalizers']:
        for neighbors in parameters['n_neighbors']:
            for metrics in parameters['metrics']:
                for weights in parameters['weights']:
                    mean_val = np.empty(len(folds))
                    for step in range(len(folds)):
                        model = knn_class(n_neighbors=neighbors, weights=weights, metric=metrics)
                        if normalizers[0] is None:
                            X_train = X[folds[step][0]]
                            X_test = X[folds[step][1]]
                        else:
                            scaler = normalizers[0]
                            scaler.fit(X[folds[step][0]])
                            X_train = scaler.transform(X[folds[step][0]])
                            X_test = scaler.transform(X[folds[step][1]])
                        model.fit(X_train, y[folds[step][0]])
                        y_predict = model.predict(X_test)
                        mean_val[step] = score_function(y[folds[step][1]], y_predict)
                    result[(normalizers[1], neighbors, metrics, weights)] = np.mean(mean_val)
    return result
