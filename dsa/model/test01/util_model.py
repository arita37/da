# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705,R0913
# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.

"""
import copy
import os
from importlib import import_module

import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix
from sklearn.decomposition import (TruncatedSVD)
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import (StratifiedKFold,
                                     cross_val_score, train_test_split)


def import_(abs_module_path, class_name=None):
    try:
        module_object = import_module(abs_module_path)
        print("imported", module_object)
        if class_name is None:
            return module_object
        target_class = getattr(module_object, class_name)
        return target_class
    except Exception as e:
        print(abs_module_path, class_name, e)


# from kmodes.kmodes import KModes
# from tabulate import tabulate


########### Dynamic Import  ONLY FOR JUPYTER #############################
# EvolutionaryAlgorithmSearchCV = import_("evolutionary_search", "EvolutionaryAlgorithmSearchCV")
esearch = import_("evolutionary_search")
lgb = import_("lightgbm")
kmodes = import_("kmodes")
catboost = import_("catboost")
tpot = import_("tpot")

##########################################################################
DIRCWD = os.getcwd()
print("os.getcwd", os.getcwd())


class dict2(object):
    def __init__(self, d):
        self.__dict__ = d


##########################################################################
def pd_dim_reduction(
        df,
        colname,
        colprefix="colsvd",
        method="svd",
        dimpca=2,
        model_pretrain=None,
        return_val="dataframe,param",
):
    """
       Dimension reduction technics
       dftext_svd, svd = pd_dim_reduction(dfcat_test, None,colprefix="colsvd",
                     method="svd", dimpca=2, return_val="dataframe,param")
    :param df:
    :param colname:
    :param colprefix:
    :param method:
    :param dimpca:
    :param return_val:
    :return:
    """
    colname = colname if colname is not None else list(df.columns)
    if method == "svd":
        if model_pretrain is None:
            svd = TruncatedSVD(n_components=dimpca, algorithm="randomized")
            svd = svd.fit(df[colname].values)
        else:
            svd = copy.deepcopy(model_pretrain)

        X2 = svd.transform(df[colname].values)
        # print(X2)
        dfnew = pd.DataFrame(X2)
        dfnew.columns = [colprefix + "_" + str(i) for i in dfnew.columns]

        if return_val == "dataframe,param":
            return dfnew, svd
        else:
            return dfnew


def sk_error(ypred, ytrue, method="r2", sample_weight=None, multioutput=None):
    from sklearn.metrics import r2_score

    if method == "rmse":
        aux = np.sqrt(np.sum((ypred - ytrue) ** 2)) / len(ytrue)
        print("Error:", aux, "Error/Stdev:", aux / np.std(ytrue))
        return aux / np.std(ytrue)

    elif method == "r2":
        r2 = r2_score(
            ytrue,
            ypred,
            sample_weight=sample_weight,
            multioutput=multioutput)
        r = np.sign(r2) * np.sqrt(np.abs(r2))
        return -1 if r <= -1 else r


######## Valuation model template  #######################################
class model_template1(sk.base.BaseEstimator):
    def __init__(self, alpha=0.5, low_y_cut=-0.09, high_y_cut=0.09, ww0=0.95):
        from sklearn.linear_model import Ridge

        self.alpha = alpha
        self.low_y_cut, self.high_y_cut, self.ww0 = 1000.0 * \
                                                    low_y_cut, 1000.0 * high_y_cut, ww0
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, Y=None):
        X, Y = X * 100.0, Y * 1000.0

        y_is_above_cut = Y > self.high_y_cut
        y_is_below_cut = Y < self.low_y_cut
        y_is_within_cut = ~y_is_above_cut & ~y_is_below_cut
        if len(y_is_within_cut.shape) > 1:
            y_is_within_cut = y_is_within_cut[:, 0]

        self.model.fit(X[y_is_within_cut, :], Y[y_is_within_cut])

        r2 = self.model.score(X[y_is_within_cut, :], Y[y_is_within_cut])
        print(("R2:", r2))
        print(("Inter", self.model.intercept_))
        print(("Coef", self.model.coef_))

        self.ymedian = np.median(Y)
        return self, r2, self.model.coef_

    def predict(self, X, y=None, ymedian=None):
        X = X * 100.0

        if ymedian is None:
            ymedian = self.ymedian
        Y = self.model.predict(X)
        Y = Y.clip(self.low_y_cut, self.high_y_cut)
        Y = self.ww0 * Y + (1 - self.ww0) * ymedian

        Y = Y / 1000.0
        return Y

    def score(self, X, Ytrue=None, ymedian=None):
        from sklearn.metrics import r2_score

        X = X * 100.0

        if ymedian is None:
            ymedian = self.ymedian
        Y = self.model.predict(X)
        Y = Y.clip(self.low_y_cut, self.high_y_cut)
        Y = self.ww0 * Y + (1 - self.ww0) * ymedian
        Y = Y / 1000.0
        return r2_score(Ytrue, Y)


def sk_model_ensemble_weight(model_list, acclevel, maxlevel=0.88):
    imax = min(acclevel, len(model_list))
    estlist = np.empty(imax, dtype=np.object)
    estww = []
    for i in range(0, imax):
        # if model_list[i,3]> acclevel:
        estlist[i] = model_list[i, 1]
        estww.append(model_list[i, 3])
        # print 5

    # Log Proba Weighted + Impact of recent False discovery
    estww = np.log(1 / (maxlevel - np.array(estww) / 2.0))
    # estww= estww/np.sum(estww)
    # return np.array(estlist), np.array(estww)
    return estlist, np.array(estww)


def sk_model_votingpredict(estimators, voting, ww, X_test):
    ww = ww / np.sum(ww)
    Yproba0 = np.zeros((len(X_test), 2))
    Y1 = np.zeros((len(X_test)))

    for k, clf in enumerate(estimators):
        Yproba = clf.predict_proba(X_test)
        Yproba0 = Yproba0 + ww[k] * Yproba

    for k in range(0, len(X_test)):
        if Yproba0[k, 0] > Yproba0[k, 1]:
            Y1[k] = -1
        else:
            Y1[k] = 1
    return Y1, Yproba0


############## ML metrics    ###################################
def sk_metric_roc_optimal_Cutoff(ytest, ytest_proba):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    ytest : Matrix with dependent or target data, where rows are observations

    ytest_proba : Matrix with predicted data, where rows are observations

    # Find prediction to the dataframe applying threshold
    data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)

    # Print confusion Matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix(data['admit'], data['pred'])
    # array([[175,  98],
    #        [ 46,  81]])

    Returns
    -------
    with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(ytest, ytest_proba)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i),
                        'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t['threshold']


def sk_showconfusion(Y, Ypred, isprint=True):
    cm = sk.metrics.confusion_matrix(Y, Ypred)
    # tn, fp, fn, tp = cm.ravel()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)

    return cm, cm_norm, (cm_norm[0, 0] + cm_norm[1, 1])


def sk_metric_roc_auc_multiclass(n_classes=3, y_test=None, y_test_pred=None, y_predict_proba=None):
    # Compute ROC curve and ROC AUC for each class
    # n_classes = 3
    conf_mat = sk.metrics.confusion_matrix(y_test, y_test_pred)
    print(conf_mat)
    if y_predict_proba is None:
        return conf_mat

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = list(map(lambda x: 1 if x == i else 0, y_test))
        # print(y_test_i)
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])

    print("auc average", roc_auc["average"])

    # Plot average ROC Curve
    plt.figure()
    plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["average"]),
             color='deeppink', linestyle=':', linewidth=4)

    # Plot each individual ROC curve
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc["average"]


def sk_showconfusion_advanced(Y, Ypred, isprint=True):
    cm = sk.metrics.confusion_matrix(Y, Ypred)
    # tn, fp, fn, tp = cm.ravel()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)

    # Create CM From Data
    cm_add = ConfusionMatrix(actual_vector=Y, predict_vector=Ypred)

    res = {"cm": cm, "cm_norm": cm_norm,
           "tp": cm[0, 0],
           "fp": cm[0, 1],
           "tn": cm[1, 1],
           "fn": cm[1, 0],
           "cm_details": cm_add}
    return res


def sk_showmetrics(y_test, ytest_pred, ytest_proba,
                   target_names=["0", "1"], return_stat=0):
    # Confusion matrix
    mtest = sk_showconfusion(y_test, ytest_pred, isprint=False)
    # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
    auc = roc_auc_score(y_test, ytest_proba)  #
    gini = 2 * auc - 1
    acc = accuracy_score(y_test, ytest_pred)
    f1macro = sk.metrics.f1_score(y_test, ytest_pred, average="macro")

    print("Test confusion matrix")
    print(mtest[0])
    print(mtest[1])
    print("auc " + str(auc))
    print("gini " + str(gini))
    print("acc " + str(acc))
    print("f1macro " + str(f1macro))
    print("Nsample " + str(len(y_test)))

    print(classification_report(y_test, ytest_pred, target_names=target_names))

    # Show roc curve
    try:
        fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(fpr, tpr, marker=".")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.show()
    except Exception as e:
        print(e)

    if return_stat:
        return {"auc": auc, "f1macro": f1macro, "acc": acc, "confusion": mtest}


def model_logistic_score(clf, df1, cols, coltarget, outype="score"):
    """

    :param clf:
    :param df1:
    :param cols:
    :param outype:
    :return:
    """

    def score_calc(yproba, pnorm=1000.0):
        yy = np.log(0.00001 + (1 - yproba) / (yproba + 0.001))
        # yy =  (yy  -  np.minimum(yy)   ) / ( np.maximum(yy) - np.minimum(yy)  )
        # return  np.maximum( 0.01 , yy )    ## Error it bias proba
        return yy

    X_all = df1[cols].values

    yall_proba = clf.predict_proba(X_all)[:, 1]
    yall_pred = clf.predict(X_all)
    try:
        y_all = df1[coltarget].values
        sk_showmetrics(y_all, yall_pred, yall_proba)
    except BaseException:
        pass

    yall_score = score_calc(yall_proba)
    yall_score = (
            1000 * (yall_score - np.min(yall_score)) /
            (np.max(yall_score) - np.min(yall_score))
    )

    if outype == "score":
        return yall_score
    if outype == "proba":
        return yall_proba, yall_pred


def sk_model_eval_regression(
        clf, istrain=1, Xtrain=None, ytrain=None, Xval=None, yval=None):
    if istrain:
        clf.fit(Xtrain, ytrain)

    CV_score = -cross_val_score(clf, Xtrain, ytrain,
                                scoring="neg_mean_absolute_error", cv=4)

    print("CV score: ", CV_score)
    print("CV mean: ", CV_score.mean())
    print("CV std:", CV_score.std())

    train_y_predicted_logReg = clf.predict(Xtrain)
    val_y_predicted_logReg = clf.predict(Xval)

    print("\n")
    print(
        "Score on logReg training set:",
        mean_absolute_error(
            ytrain,
            train_y_predicted_logReg))
    print(
        "Score on logReg validation set:",
        mean_absolute_error(
            yval,
            val_y_predicted_logReg))

    return clf, train_y_predicted_logReg, val_y_predicted_logReg


def sk_model_eval_classification(
        clf, istrain=1, Xtrain=None, ytrain=None, Xtest=None, ytest=None):
    if istrain:
        print("############# Train dataset  ####################################")
        clf.fit(Xtrain, ytrain)
        ytrain_proba = clf.predict_proba(Xtrain)[:, 1]
        ytrain_pred = clf.predict(Xtrain)
        # sk_showmetrics(ytrain, ytrain_pred, ytrain_proba)

    print("############# Test dataset  #########################################")
    ytest_proba = clf.predict_proba(Xtest)[:, 1]
    ytest_pred = clf.predict(Xtest)
    # sk_showmetrics(ytest, ytest_pred, ytest_proba)

    return clf, {"ytest_pred": ytest_pred, "ytest_proba": ytest_proba}


def sk_model_eval(
        clf, istrain=1, Xtrain=None, ytrain=None, Xtest=None, ytest=None):
    if istrain:
        print("############# Train dataset  ####################################")
        clf.fit(Xtrain, ytrain)
        ytrain_proba = clf.predict_proba(Xtrain)[:, 1]
        ytrain_pred = clf.predict(Xtrain)
        # sk_showmetrics(ytrain, ytrain_pred, ytrain_proba)

    print("############# Test dataset  #########################################")
    ytest_proba = clf.predict_proba(Xtest)[:, 1]
    ytest_pred = clf.predict(Xtest)
    # sk_showmetrics(ytest, ytest_pred, ytest_proba)

    res = {"ytest_pred": ytest_pred, "ytest_proba": ytest_proba,
           "ytrain_pred": ytrain_pred, "ytrain_proba": ytrain_proba}

    return clf, res


##########################################################################
def sk_feature_impt(clf, colname, model_type="logistic"):
    """
       Feature importance with colname
    :param clf:  model or colnum with weights
    :param colname:
    :return:
    """
    if model_type == "logistic":
        dfeatures = pd.DataFrame(
            {"feature": colname, "weight": clf.coef_[
                0], "weight_abs": np.abs(clf.coef_[0])}
        ).sort_values("weight_abs", ascending=False)
        dfeatures["rank"] = np.arange(0, len(dfeatures))
        return dfeatures

    else:
        # RF, Xgboost, LightGBM
        if isinstance(clf, list) or isinstance(clf, (np.ndarray, np.generic)):
            importances = clf
        else:
            importances = clf.feature_importances_
        rank = np.argsort(importances)[::-1]
        d = {"col": [], "rank": [], "weight": []}
        for i in range(0, len(colname)):
            d["rank"].append(rank[i])
            d["col"].append(colname[rank[i]])
            d["weight"].append(importances[rank[i]])

        return pd.DataFrame(d)


def sk_feature_selection(clf, method="f_classif",
                         colname=None, kbest=50, Xtrain=None, ytrain=None):
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    if method == "f_classif":
        clf_best = SelectKBest(f_classif, k=kbest).fit(Xtrain, ytrain)

    if method == "f_regression":
        clf_best = SelectKBest(f_regression, k=kbest).fit(Xtrain, ytrain)

    mask = clf_best.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, colname):
        if bool:
            new_features.append(feature)

    return new_features


def sk_feature_evaluation(clf, df, kbest=30, colname_best=None, dfy=None):
    clf2 = copy.deepcopy(clf)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        df[colname_best].values, dfy.values, random_state=42, test_size=0.5, shuffle=True
    )
    print(Xtrain.shape, ytrain.shape)

    df = {x: [] for x in ["col", "auc", "acc", "f1macro", "confusion"]}
    for i in range(1, len(colname_best)):
        print("########## ", colname_best[:i])
        if i > kbest:
            break
        clf.fit(Xtrain[:, :i], ytrain)
        ytest_proba = clf.predict_proba(Xtest[:, :i])[:, 1]
        ytest_pred = clf.predict(Xtest[:, :i])
        s = sk_showmetrics(ytest, ytest_pred, ytest_proba, return_stat=1)

        # {"auc": auc, "f1macro": f1macro, "acc": acc, "confusion": mtest}

        df["col"].append(str(colname_best[:i]))
        df["auc"].append(s["auc"])
        df["acc"].append(s["acc"])
        df["f1macro"].append(s["f1macro"])
        df["confusion"].append(s["confusion"])

    df = pd.DataFrame(df)
    return df


def sk_feature_drift_covariance(dftrain, dftest, colname, nsample=10000):
    n1 = nsample if len(dftrain) > nsample else len(dftrain)
    n2 = nsample if len(dftest) > nsample else len(dftest)
    train = dftrain[colname].sample(n1, random_state=12)
    test = dftest[colname].sample(n2, random_state=11)

    # creating a new feature origin
    train["origin"] = 0
    test["origin"] = 1

    # combining random samples
    combi = train.append(test)
    y = combi["origin"]
    combi.drop("origin", axis=1, inplace=True)

    # modelling
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        min_samples_leaf=5)
    drop_list = []
    for i in combi.columns:
        score = cross_val_score(
            model, pd.DataFrame(
                combi[i]), y, cv=2, scoring="roc_auc")

        if np.mean(score) > 0.8:
            drop_list.append(i)
        print(i, np.mean(score))
    return drop_list


def sk_model_eval_classification_cv(
        clf, X, y, test_size=0.5, ncv=1, method="random"):
    """
    :param clf:
    :param X:
    :param y:
    :param test_size:
    :param ncv:
    :param method:
    :return:
    """
    if method == "kfold":
        kf = StratifiedKFold(n_splits=ncv, shuffle=True)
        clf_list = {}
        for i, itrain, itest in enumerate(kf.split(X, y)):
            print("###")
            Xtrain, Xtest = X[itrain], X[itest]
            ytrain, ytest = y[itrain], y[itest]
            clf_list[i], _ = sk_model_eval_classification(
                clf, 1, Xtrain, ytrain, Xtest, ytest)

    else:
        clf_list = {}
        for i in range(0, ncv):
            print(
                "############# CV-{i}######################################".format(i=i))
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X, y, test_size=test_size, shuffle=True)

            clf_list[i], _ = sk_model_eval_classification(
                clf, 1, Xtrain, ytrain, Xtest, ytest)

    return clf_list


def sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base=" "):
    """Produce psuedo-code for decision tree.
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (output) names.
    spacer_base -- used for spacing code (default: "    ").
    """
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if threshold[node] != -2:
            print((spacer +
                   "if " +
                   features[node] +
                   " <= " +
                   str(threshold[node]) +
                   " :"))
            #            print(spacer + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) :")
            if left[node] != -1:
                recurse(
                    left,
                    right,
                    threshold,
                    features,
                    left[node],
                    depth + 1)
            print(("" + spacer + "else :"))
            if right[node] != -1:
                recurse(
                    left,
                    right,
                    threshold,
                    features,
                    right[node],
                    depth + 1)
        #     print(spacer + "")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(
                    (
                            spacer
                            + "return "
                            + str(target_name)
                            + " ( "
                            + str(target_count)
                            + ' examples )"'
                    )
                )

    recurse(left, right, threshold, features, 0, 0)
