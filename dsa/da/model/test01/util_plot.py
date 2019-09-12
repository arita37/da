# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705,R0913
"""
Methods for data plotting
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sci
import seaborn as sns
import sklearn as sk

try:
    import plotly
    import cufflinks as cf
except Exception as e:
    print(e)


##########################################################################


def plotbar(df, colname, figsize=(20, 10),
            title="feature importance", savefile="myfile.png"):
    plt.figure(figsize=(20, 10))
    sns.barplot(x=colname[0], y=colname[1], data=df[colname])
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig(savefile)


def plotxy(
        x, y, color=1, size=1, figsize=(12, 10), title="feature importance", savefile="myfile.png"
):
    """
    :param x:
    :param y:
    :param color:
    :param size:
    :param title:
    """

    color = np.zeros(len(x)) if isinstance(color, int) else color
    fig, ax = plt.subplots(figsize=figsize)
    plt.scatter(x, y, c=color, cmap="Spectral", s=size)
    plt.title(title, fontsize=11)
    plt.show()
    plt.savefig(savefile)


def plot_col_distribution(df, col_include=None,
                          col_exclude=None, pars={"binsize": 20}):
    """  Retrives all the information of the column
    :param df:
    :param col_include:
    :param col_exclude:
    :param pars:
    """
    features = list()

    if col_include is not None:
        features = [
            feature for feature in df.columns.values if feature in col_include]

    elif col_exclude is not None:
        features = [
            feature for feature in df.columns.values if not feature in col_exclude]
    elif col_exclude is None and col_include is None:
        features = [feature for feature in df.columns.values]

    for feature in features:
        values = df[feature].values
        nan_count = np.count_nonzero(np.isnan(values))
        values = sorted(values[~np.isnan(values)])
        print(("NaN count:", nan_count, "Unique count:", len(np.unique(values))))
        print(("Max:", np.max(values), "Min:", np.min(values)))
        print(("Median", np.median(values), "Mean:",
               np.mean(values), "Std:", np.std(values)))
        plot_Y(
            values,
            typeplot=".b",
            title="Values " +
                  feature,
            figsize=(
                8,
                5))

        fit = sci.stats.norm.pdf(
            values, np.mean(values), np.std(values)
        )  # this is a fitting indeed
        plt.title("Distribution Values " + feature)
        plt.plot(values, fit, "-g")
        plt.hist(
            values, normed=True, bins=pars["binsize"]
        )  # use this to draw histogram of your data
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.title("Percentiles 5...95" + feature)
        plt.plot(
            list(
                range(
                    5, 100, 5)), np.percentile(
                values, list(
                    range(
                        5, 100, 5))), ".b")
        plt.show()


def plot_pair(df, Xcolname=None, Ycoltarget=None):
    """
    :param df:
    :param Xcolname:
    :param Ycoltarget:

    """
    yy = df[Ycoltarget].values

    for coli in Xcolname:
        xx = df[coli].values
        title1 = "X: " + str(coli) + ", Y: " + str(Ycoltarget[0])
        plt.scatter(xx, yy, s=1)
        plt.autoscale(enable=True, axis="both", tight=None)
        #  plt.axis([-3, 3, -3, 3])  #gaussian
        plt.title(title1)
        plt.show()


def plot_distance_heatmap(Xmat_dist, Xcolname):
    """

    :param Xmat_dist:
    :param Xcolname:
    :return:
    """
    """
    :param Xmat_dist:
    :param Xcolname:
    :return:
    """
    import matplotlib.pyplot as pyplot

    df = pd.DataFrame(Xmat_dist)
    df.columns = Xcolname
    df.index.name = "Col X"
    df.columns.name = "Col Y"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(
        df.values,
        cmap=pyplot.get_cmap("RdYlGn"),
        interpolation="nearest")
    ax.set_xlabel(df.columns.name)
    ax.set_ylabel(df.index.name)
    ax.set_title("Pearson R Between Features")
    plt.colorbar(axim)


def plot_cluster_2D(X_2dim, target_class, target_names):
    """
    :param X_2dim:
    :param target_class:
    :param target_names:
    :return:
    Plot 2d of Clustering Class,
    X2d: Nbsample x 2 dim  (projection on 2D sub-space)
   """
    colors = itertools.cycle("rgbcmykw")
    target_ids = range(0, len(target_names))
    plt.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2dim[target_class == i, 0],
                    X_2dim[target_class == i, 1], c=c, label=label)
    plt.legend()
    plt.show()


def plot_cluster_tsne(
        Xmat,
        Xcluster_label=None,
        metric="euclidean",
        perplexity=50,
        ncomponent=2,
        savefile="",
        isprecompute=False,
        returnval=True,
):
    """
    :return:

    Plot High dimemnsionnal State using TSNE method
   'euclidean, 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean, 'cosine, 'correlation, 'hamming, 'jaccard, 'chebyshev,
   'canberra, 'braycurtis, 'mahalanobis', VI=None) 'yule, 'matching, 'dice, 'kulsinski, 'rogerstanimoto, 'russellrao, 'sokalmichener, 'sokalsneath,

   Xtsne= da.plot_cluster_tsne(Xtrain_dist, Xcluster_label=None, perplexity=40, ncomponent=2, isprecompute=True)

   Xtrain_dist= sci.spatial.distance.squareform(sci.spatial.distance.pdist(Xtrain_d,
               metric='cityblock', p=2, w=None, V=None, VI=None))
   """

    if isprecompute:
        Xmat_dist = Xmat
    else:
        Xmat_dist = sci.spatial.distance.squareform(
            sci.spatial.distance.pdist(
                Xmat, metric=metric, p=ncomponent, w=None, V=None, VI=None)
        )

    model = sk.manifold.TSNE(
        n_components=ncomponent, perplexity=perplexity, metric="precomputed", random_state=0
    )
    np.set_printoptions(suppress=True)
    X_tsne = model.fit_transform(Xmat_dist)

    # plot the result
    xx, yy = X_tsne[:, 0], X_tsne[:, 1]
    if Xcluster_label is None:
        Yclass = np.arange(0, X_tsne.shape[0])
    else:
        Yclass = Xcluster_label

    plot_XY(
        xx,
        yy,
        zcolor=Yclass,
        labels=Yclass,
        color_dot="plasma",
        savefile=savefile)

    if returnval:
        return X_tsne


def plot_cluster_pca(
        Xmat,
        Xcluster_label=None,
        metric="euclidean",
        dimpca=2,
        whiten=True,
        isprecompute=False,
        savefile="",
        doreturn=1,
):
    """
    :return:
    """

    from sklearn.decomposition import pca

    if isprecompute:
        Xmat_dist = Xmat
    else:
        Xmat_dist = sci.spatial.distance.squareform(
            sci.spatial.distance.pdist(
                Xmat, metric=metric, p=dimpca, w=None, V=None, VI=None)
        )

    model = pca(n_components=dimpca, whiten=whiten)
    X_pca = model.fit_transform(Xmat)

    # plot the result
    xx, yy = X_pca[:, 0], X_pca[:, 1]
    if Xcluster_label is None:
        Yclass = np.zeros(X_pca.shape[0])
    else:
        Yclass = Xcluster_label

    plot_XY(
        xx,
        yy,
        zcolor=Yclass,
        labels=Yclass,
        color_dot="plasma",
        savefile=savefile)

    if doreturn:
        return X_pca


def plot_cluster_hiearchy(
        Xmat_dist,
        p=30,
        truncate_mode=None,
        color_threshold=None,
        get_leaves=True,
        orientation="top",
        labels=None,
        count_sort=False,
        distance_sort=False,
        show_leaf_counts=True,
        do_plot=1,
        no_labels=False,
        leaf_font_size=None,
        leaf_rotation=None,
        leaf_label_func=None,
        show_contracted=False,
        link_color_func=None,
        ax=None,
        above_threshold_color="b",
        annotate_above=0,
):
    """
    :return:
    """
    from scipy.cluster.hierarchy import dendrogram

    ddata = dendrogram(
        Xmat_dist,
        p=30,
        truncate_mode=truncate_mode,
        color_threshold=color_threshold,
        get_leaves=get_leaves,
        orientation="top",
        labels=None,
        count_sort=False,
        distance_sort=False,
        show_leaf_counts=True,
        no_plot=1 - do_plot,
        no_labels=False,
        leaf_font_size=None,
        leaf_rotation=None,
        leaf_label_func=None,
        show_contracted=False,
        link_color_func=None,
        ax=None,
        above_threshold_color="b",
    )

    if do_plot:
        annotate_above = 0
        plt.title("Hierarchical Clustering Dendrogram (truncated)")
        plt.xlabel("sample index or (sk_cluster size)")
        plt.ylabel("distance")
        for i, d, c in zip(
                ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, "o", c=c)
                plt.annotate(
                    "%.3g" % y,
                    (x, y),
                    xytext=(0, -5),
                    textcoords="offset points",
                    va="top",
                    ha="center",
                )
        if color_threshold:
            plt.axhline(y=color_threshold, c="k")
    return ddata


def plot_distribution_density(
        Xsample, kernel="gaussian", N=10, bandwith=1 / 10.0):
    """ from scipy.optimize import brentq
import statsmodels.api as sm
import numpy as np

# fit
kde = sm.nonparametric.KDEMultivariate()  # ... you already did this

# sample
u = np.random.random()

# 1-d root-finding
def func(x):
    return kde.cdf([x]) - u
sample_x = brentq(func, -99999999, 99999999)  # read brentq-docs about these constants
                                              # constants need to be sign-changing for the function
  """

    fig, ax = plt.subplots()
    XN = len(Xsample)
    xmin, xmax = np.min(Xsample), np.max(Xsample)
    X_plot = np.linspace(xmin, xmax, XN)[:, np.newaxis]
    bins = np.linspace(xmin, xmax, N)

    # Xhist, Xbin_edges= np.histogram(Xsample, bins=bins, range=None, normed=False, weights=None, density=True)

    weights = np.ones_like(Xsample) / len(Xsample)  # np.ones(len(Xsample))  #
    # ax2.hist(ret5d,50, normed=0,weights=weights,  facecolor='green')
    ax.hist(Xsample, bins=N, normed=0, weights=weights, fc="#AAAAFF")

    kde = sk.neighbors.KernelDensity(
        kernel=kernel, bandwidth=bandwith).fit(Xsample.reshape(-1, 1))
    log_dens = kde.score_samples(X_plot)
    log_dens -= np.log(XN)  # Normalize
    ax.plot(X_plot[:, 0], np.exp(log_dens), "-",
            label="kernel = '{0}'".format(kernel))

    ax.set_xlim(xmin, xmax)
    plt.show()
    return kde


def plot_Y(
        Yval,
        typeplot=".b",
        tsize=None,
        labels=None,
        title="",
        xlabel="",
        ylabel="",
        zcolor_label="",
        figsize=(8, 6),
        dpi=75,
        savefile="",
        color_dot="Blues",
        doreturn=0,
):
    """
     Return plot values
    """
    plt.figure(figsize=figsize)
    plt.title("Values " + title)
    plt.plot(Yval, typeplot)
    plt.show()


def plot_XY(
        xx,
        yy,
        zcolor=None,
        tsize=None,
        labels=None,
        title="",
        xlabel="",
        ylabel="",
        zcolor_label="",
        figsize=(8, 6),
        dpi=75,
        savefile="",
        color_dot="Blues",
        doreturn=0,
):
    """
      labels= numpy array, ---> Generate HTML File with the labels interactives
      Color: Plasma
    """

    # Color change
    if zcolor is None:
        c = [[0, 0, 0]]
    elif isinstance(zcolor, int):
        zcolor = zcolor
    else:
        aux = np.array(zcolor, dtype=np.float64)
        c = np.abs(aux)
    cmhot = plt.get_cmap(color_dot)

    # Marker size
    if tsize is None:
        tsize = 50
    elif isinstance(tsize, int):
        tsize = tsize
    else:
        aux = np.array(tsize, dtype=np.float64)
        tsize = np.abs(aux)
        tsize = (tsize - np.min(tsize)) / \
                (np.max(tsize) - np.min(tsize)) * 130 + 1

    # Plot
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # Overall Plot
    fig.set_size_inches(figsize[0], figsize[1])
    fig.set_dpi(dpi)
    fig.tight_layout()

    # Scatter
    scatter = ax1.scatter(xx, yy, c=c, cmap=cmhot, s=tsize, alpha=0.5)
    ax1.set_xlabel(xlabel, fontsize=9)
    ax1.set_ylabel(ylabel, fontsize=9)
    ax1.set_title(title)
    ax1.grid(True)
    # fig.autoscale(enable=True, axis='both')
    # fig.colorbar(ax1)

    c_min, c_max = np.min(c), np.max(c)
    scatter.set_clim([c_min, c_max])
    cb = fig.colorbar(scatter)
    cb.set_label(zcolor_label)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # cax = ax1.imshow(c, interpolation='nearest', cmap=color_dot)

    # cbar = fig.colorbar(ax1, ticks= xrange(c_min, c_max, 10))
    # cbar.ax.set_yticklabels([str(c_min), str(c_max)])  # vertically oriented colorbar
    # plt.clim(-0.5, 9.5)

    #if labels is not None:  # Interactive HTML
    #    import mpld3
    #
    #    labels = list(labels)
    #    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    #    mpld3.plugins.connect(fig, tooltip)
    #    mpld3.save_html(fig, savefile + ".html")

    plt.show()
    if savefile != "":
        plt.savefig(savefile)

    if doreturn:
        return fig, ax1


def plot_XY_seaborn(X, Y, Zcolor=None):
    """
    :param X:
    :param Y:
    :param Zcolor:
    :return:
    """
    sns.set_context("poster")
    sns.set_color_codes()
    plot_kwds = {"alpha": 0.35, "s": 60, "linewidths": 0}
    palette = sns.color_palette("deep", np.unique(Zcolor).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in Zcolor]
    plt.scatter(X, Y, c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title("X:   , Y:   ,Z:", fontsize=18)


# Added functions
######################################


def plot_col_correl_matrix(df, cols, annot=True, size=30):
    """
    Function to plot correlation matrix
    Arguments:
        df:    dataframe
        cols:  columns to correlate
        annot: annotate or not (default = True)
        size:  size of correlation matrix (default = 30)
    Prints:
        correlation matrix of columns to each other
    """
    sns.heatmap(df[cols].corr(), cmap="coolwarm", annot=annot).set_title(
        "Correlation Matrix", size=size
    )
