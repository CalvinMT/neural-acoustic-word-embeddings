from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_expectations():
    """
    Somehow compute and return expected results.

    :return: array of expected results
    """
    expectations = []

    # TODO - compute expectations

    return expectations


def get_confusion_matrix(results, expectations):
    """
    Compare results with expectations and return the number of true positive (TP),
    true negative (TN), false positive (FP), false negative (FN) respectively.
    The results array and the expectations array don't have to be of the same length,
    though the results array's length have to be equal or higher than the expectations
    array's length.

    :param results: array of obtained results
    :param expectations: array of expected results
    :return: tuple of four floats
    """
    assert len(results) >= len(expectations)
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0

    # TODO - compute TP, TN, FP and FN out of the comparison between results and expectations

    return TP, TN, FP, FN

def get_TPR_and_FPR(TP, TN, FP, FN):
    """
    Compute and return the true positive rate (TPR) and the false positive rate (FPR)
    from the number of true positive (TP), true negative (TN), false positive (FP),
    false negative (FN) respectively.

    :param TP: float
    :param TN: float
    :param FP: float
    :param FN: float
    :return: tuple of two floats
    """
    TPR = 0.0       # recall
    FPR = 0.0       # fall-out
    if TP > 0.0:
        TPR = TP / (TP + FN)
    if TN > 0.0:
        FPR = TN / (TN + FP)
    return TPR, FPR


def compute_results(dists, threshold):
    """
    Compare sweeping scores with the given threshold. The result is 0 if the score is 
    strictly higher than the threshold; 1 otherwise.

    :param dists: distance between embeddings
    :param threshold: float between 0.0 and 1.0
    :return: array of obtained results
    """
    results = []
    for d in dists:
        if d <= threshold:
            results.append(1)
        else:
            results.append(0)
    return results


def compute_ROC_curve(embeddings_x, embeddings_y, nb_thresholds=100):
    """
    Compute ROC points for each threshold (from 0 to nb_thresholds-1) and return them
    as a pandas DataFrame, along with the Area Under the Curve (AUC).

    :param embeddings_x: array representing embeddings
    :param embeddings_y: array representing embeddings
    :param nb_thresholds: integer (number of thresholds)
    :return: AUC, ROC curve pivots (x, y, thresholds)
    """
    dists = cdist(embeddings_x, embeddings_y, metric="cosine")

    expectations = get_expectations()

    threshold_list = list(np.array(list(range(0, nb_thresholds + 1, 1))) / float(nb_thresholds))
    roc_points = []

    for threshold in threshold_list:
        results = compute_results(dists, threshold)
        TP, TN, FP, FN = get_confusion_matrix(results, expectations)
        TPR, FPR = get_TPR_and_FPR(TP, TN, FP, FN)
        roc_points.append([TPR, FPR])
    
    pivot = pd.DataFrame(roc_points, columns=["x", "y"])
    pivot["thresholds"] = threshold_list

    AUC = abs(np.trapz(pivot.x, pivot.y))

    return AUC, pivot
