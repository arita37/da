# -*- coding: utf-8 -*-


from scipy import interp
from sklearn.metrics import roc_curve, auc


def _display_plot():
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def roc_graph_binary(y_true, y_pred, **kwargs):
    """
    This function plots a ROC graph of a binary-class predictor. AUC calculation are presented as-well.
    Data can be either: (1) one dimensional, where the values of y_true represent the true class and y_pred the
    predicted probability of that class, or (2) two-dimensional, where each line in y_true is a one-hot-encoding
    of the true class and y_pred holds the predicted probabilities of each class.
    For example, consider a data-set of two data-points where the true class of the first line is class 0, which
    was predicted with a probability of 0.6, and the second line's true class is 1, with predicted probability of
    0.8. In the first configuration, the input will be: y_true = [0,1], y_pred = [0.6,0.8]. In the second
    configuration, the input will be: y_true = [[1,0],[0,1]], y_pred = [[0.6,0.4],[0.2,0.8]].
    Based on sklearn examples (as was seen on April 2018):
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    Parameters
    ----------
    y_true : list / NumPy ndarray
        The true classes of the predicted data
    y_pred : list / NumPy ndarray
        The predicted classes
    kwargs : any key-value pairs
        Different options and configurations
    """
    y_true = convert(y_true, "array")
    y_pred = convert(y_pred, "array")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    elif len(y_pred.shape) == 1:
        y_t = y_true
        y_p = y_pred
    else:
        y_t = [np.argmax(x) for x in y_true]
        y_p = [x[1] for x in y_pred]
    fpr, tpr, _ = roc_curve(y_t, y_p)
    auc_score = auc(fpr, tpr)
    color = kwargs.get("color", "darkorange")
    lw = kwargs.get("lw", 2)
    ls = kwargs.get("ls", "-")
    ms = kwargs.get("ms", 10)
    fmt = kwargs.get("fmt", ".2f")
    if "class_label" in kwargs:
        class_label = ": {}".format(kwargs["class_label"])
    else:
        class_label = ""
    if kwargs.get("new_figure", True):
        plt.figure()
    plt.plot(
        fpr,
        tpr,
        color=color,
        lw=lw,
        ls=ls,
        label="ROC curve{class_label} (AUC = {auc:{fmt}})".format(
            class_label=class_label, auc=auc_score, fmt=fmt
        ),
    )
    if kwargs.get("show_graphs", True):
        _display_plot()
    if kwargs.get("return_pr", False):
        return {"fpr": fpr, "tpr": tpr}


def _plot_macro_roc(fpr, tpr, n, **kwargs):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    auc_macro = auc(fpr_macro, tpr_macro)
    fmt = kwargs.get("fmt", ".2f")
    lw = kwargs.get("lw", 2)
    plt.plot(
        fpr_macro,
        tpr_macro,
        label="ROC curve: macro (AUC = {auc:{fmt}})".format(auc=auc_macro, fmt=fmt),
        color="navy",
        ls=":",
        lw=lw,
    )


def roc_graph(y_true, y_pred, micro=True, macro=True, **kwargs):
    """
    Plot a ROC graph of predictor's results (inclusding AUC scores), where each row of y_true and y_pred
    represent a single example.
    If there are 1 or two columns only, the data is treated as a binary classification, in which
    the result is similar to the `binary_roc_graph` method, see its documentation for more information.
    If there are more then 2 columns, each column is considered a unique class, and a ROC graph and AUC
    score will be computed for each. A Macro-ROC and Micro-ROC are computed and plotted too by default.
    Based on sklearn examples (as was seen on April 2018):
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    **Example:** See `roc_graph_example` under `dython.examples`
    Parameters
    ----------
    y_true : list / NumPy ndarray
        The true classes of the predicted data
    y_pred : list / NumPy ndarray
        The predicted classes
    micro : Boolean, default = True
        Whether to calculate a Micro ROC graph (not applicable for binary cases)
    macro : Boolean, default = True
        Whether to calculate a Macro ROC graph (not applicable for binary cases)
    kwargs : any key-value pairs
        Different options and configurations
    """
    all_fpr = list()
    all_tpr = list()
    y_true = convert(y_true, "array")
    y_pred = convert(y_pred, "array")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    elif len(y_pred.shape) == 1 or y_pred.shape[1] <= 2:
        return binary_roc_graph(y_true, y_pred, **kwargs)
    else:
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        n = y_pred.shape[1]
        plt.figure()
        kwargs["new_figure"] = False
        kwargs["show_graphs"] = False
        kwargs["return_pr"] = True
        for i in range(0, n):
            pr = binary_roc_graph(
                y_true[:, i], y_pred[:, i], color=colors[i % len(colors)], class_label=i, **kwargs
            )
            all_fpr.append(pr["fpr"])
            all_tpr.append(pr["tpr"])
        if micro:
            binary_roc_graph(
                y_true.ravel(),
                y_pred.ravel(),
                ls=":",
                color="deeppink",
                class_label="micro",
                **kwargs
            )
        if macro:
            _plot_macro_roc(all_fpr, all_tpr, n)
        _display_plot()


def feature_importance_rf(clf, features, **kwargs):
    """
    Given a trained `sklearn.ensemble.RandomForestClassifier`, plot the different features based on their
    importance according to the classifier, from the most important to the least.
    Parameters
    ----------
    clf : sklearn.ensemble.RandomForestClassifier
        A trained `RandomForestClassifier`
    features : list
        A list of the names of the features the classifier was trained on, ordered by the same order the appeared
        in the training data
    kwargs : any key-value pairs
        Different options and configurations
    """
    return sorted(
        zip(
            map(lambda x: round(x, kwargs.get("precision", 4)), clf.feature_importances_), features
        ),
        reverse=True,
    )


def convert(data, to):
    converted = None
    if to == "array":
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == "list":
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == "dataframe":
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError("cannot handle data conversion of type: {} to {}".format(type(data), to))
    else:
        return converted


import math
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
from collections import Counter

# from dython._private import convert


import numpy as np


def np_sampling_weighted(numbers, k=1, with_replacement=False, **kwargs):
    """
    Return k numbers from a weighted-sampling over the supplied numbers
    **Returns:** List, np.ndarray or a single number (depending on the input)
    Parameters
    ----------
    numbers : List or np.ndarray
        numbers to sample
    k : int, default = 1
        How many numbers to sample. Choosing `k=None` will yield a single number
    with_replacement : Boolean, default = False
        Allow replacement or not
    """
    sampled = np.random.choice(numbers, size=k, replace=with_replacement)
    if (isinstance(numbers, list) or kwargs.get("to_list", False)) and k is not None:
        sampled = sampled.tolist()
    return sampled


def np_sampling_boltzmann(numbers, k=1, with_replacement=False):
    """
    Return k numbers from a boltzmann-sampling over the supplied numbers
    **Returns:** List, np.ndarray or a single number (depending on the input)
    Parameters
    ----------
    numbers : List or np.ndarray
        numbers to sample
    k : int, default = 1
        How many numbers to sample. Choosing `k=None` will yield a single number
    with_replacement : Boolean, default = False
        Allow replacement or not
    """
    exp_func = np.vectorize(lambda x: np.exp(x))
    exp_numbers = exp_func(numbers)
    exp_sum = exp_numbers.sum()
    scaling_func = np.vectorize(lambda x: x / exp_sum)
    b_numbers = scaling_func(exp_numbers)
    return np_weighted_sampling(
        b_numbers, k=k, with_replacement=with_replacement, to_list=isinstance(numbers, list)
    )
