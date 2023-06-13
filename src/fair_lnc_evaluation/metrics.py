import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import mlflow
from fairlearn.metrics import demographic_parity_difference

def predictive_equality_difference(y_true:pd.Series, y_pred:pd.Series, sensitive_attr:pd.Series):
    """
    Predictive equality difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Predictive equality difference
    """
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(fp_rate(y_true_0, y_pred_0) - fp_rate(y_true_1, y_pred_1))

def equal_opportunity_difference(y_true:pd.Series, y_pred:pd.Series, sensitive_attr:pd.Series):
    """
    Equal opportunity difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Equal opportunity difference
    """
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(fn_rate(y_true_0, y_pred_0) - fn_rate(y_true_1, y_pred_1))

def tp_rate(y_true:pd.Series, y_pred:pd.Series):
    """
    True positive rate

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels

    Returns
    -------
    float
        True positive rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)

def fp_rate(y_true:pd.Series, y_pred:pd.Series):
    """
    False positive rate

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels

    Returns
    -------
    float
        False positive rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fp + tn) == 0:
        return 0
    return fp / (fp + tn)

def fn_rate(y_true:pd.Series, y_pred:pd.Series):
    """
    False negative rate

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels

    Returns
    -------
    float
        False negative rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fn + tp) == 0:
        return 0
    return fn / (fn + tp)

def eq_odds_difference(y_true:pd.Series, y_pred:pd.Series, sensitive_attr:pd.Series):
    """
    Equalized odds difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Equalized odds difference
    """
    # TPR difference

    tpr_0 = tp_rate(y_true.loc[sensitive_attr == 0], y_pred.loc[sensitive_attr == 0])
    tpr_1 = tp_rate(y_true.loc[sensitive_attr == 1], y_pred.loc[sensitive_attr == 1])
    tpr_diff = abs(tpr_0 - tpr_1)

    # FPR difference
    fpr_0 = fp_rate(y_true.loc[sensitive_attr == 0], y_pred.loc[sensitive_attr == 0])
    fpr_1 = fp_rate(y_true.loc[sensitive_attr == 1], y_pred.loc[sensitive_attr == 1])
    fpr_diff = abs(fpr_0 - fpr_1)

    return max(tpr_diff, fpr_diff)

def auc_difference(y_true:pd.Series, y_pred_proba:pd.Series, sensitive_attr:pd.Series):
    """
    AUC difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred_proba : pd.Series
        Predicted probabilities
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        AUC difference
    """
    try:
        auc_0 = roc_auc_score(y_true.loc[sensitive_attr == 0], y_pred_proba.loc[sensitive_attr == 0])
    except ValueError:
        auc_0 = 0
    try:
        auc_1 = roc_auc_score(y_true.loc[sensitive_attr == 1], y_pred_proba.loc[sensitive_attr == 1])
    except ValueError:
        auc_1 = 0

    return auc_0 - auc_1


def reconstruction_score(y:pd.Series, y_train_corrected:pd.Series, y_test_corrected:pd.Series):
    """
    Evaluate the similarity of the corrected labels to the original ones

    Parameters
    ----------
    y : pd.Series
        Original labels
    y_train_corrected : pd.Series
        Corrected training labels
    y_test_corrected : pd.Series
        Corrected test labels

    Returns
    -------
    float
        Reconstruction score
    """
    original_labels = y.sort_index()
    corrected_labels = pd.concat([y_train_corrected, y_test_corrected]).sort_index()
    
    r = accuracy_score(original_labels.values, corrected_labels.values)

    return r

def calculate_metric(metric, y_test:pd.Series, y_pred_proba, sensitive_attr:pd.Series, thresh=0.5):
    if metric == 'roc_auc':
        return roc_auc_score(y_test, y_pred_proba)
    elif metric == 'auc_difference':
        return auc_difference(y_test, y_pred_proba, sensitive_attr)

    y_pred = pd.Series(np.where(y_pred_proba > thresh, 1, 0), index=y_test.index)

    if metric == 'accuracy':
        return accuracy_score(y_test, y_pred)
    elif metric == 'demographic_parity_difference':
        return demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_attr)
    elif metric == 'equalized_odds_difference':
        return eq_odds_difference(y_test, y_pred, sensitive_attr)
    elif metric == 'predictive_equality_difference':
        return predictive_equality_difference(y_test, y_pred, sensitive_attr)
    elif metric == 'equal_opportunity_difference':
        return equal_opportunity_difference(y_test, y_pred, sensitive_attr)
    else:
        raise ValueError(f'Unknown metric: {metric}')

def log_metrics(y_test:pd.Series, y_pred_proba:pd.Series, sensitive_attr:pd.Series, metrics:list, classification_thresholds:list=[0.2, 0.5, 0.8]):
    """
    Calculate and log evaluation metrics to MLflow

    Parameters
    ----------
    y_test : pd.Series
        True labels
    y_pred_proba : pd.Series
        Predicted positive label probabilities
    sensitive_attr : pd.Series
        Sensitive attribute
    metrics : list
        List of metrics to calculate
    classification_thresholds : list, optional
        List of classification thresholds to consider, by default [0.2, 0.5, 0.8]
    """
    for metric in metrics:
        if metric == 'roc_auc' or metric == 'auc_difference':
            if len(set(y_pred_proba)) > 1 and len(y_test.unique()) > 1:
                mlflow.log_metric(metric, calculate_metric(metric, y_test, y_pred_proba, sensitive_attr))
        
        else:
            for thresh in classification_thresholds:
                mlflow.log_metric(f"{metric}_{thresh}", calculate_metric(metric, y_test, y_pred_proba, sensitive_attr, thresh))