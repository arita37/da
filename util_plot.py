"""
Methods for data plotting
"""

def plotxy(x,y, color=1, size=1, title= "") :
    pass
#  color = np.zeros(len(x)) if type(color) == int else color  
#  fig, ax = plt.subplots(figsize=(12, 10))
#  plt.scatter( x , y,  c= color, cmap="Spectral", s=size)
#  plt.title(   title, fontsize=11 )
#  plt.show()

#### Histo     
def np_histo(dfm2, bins=50, col0='diff', col1='y') :
    pass

def pd_col_study_distribution_show(df, col_include=None, col_exclude=None, pars={"binsize": 20}):
    pass

def pd_col_pair_plot(dfX, Xcolname_selectlist=None, dfY=None, Ycolname=None):
    pass

# ??? for what if pd_col_pair_plot exist? 
def plot_col_pair(dfX, Xcolname_selectlist=None, dfY=None, Ycolname=None):
    pass

def plot_distance_heatmap(Xmat_dist, Xcolname):
    pass


def plot_cluster_2D(X_2dim, target_class, target_names):
    """ Plot 2d of Clustering Class,
       X2d: Nbsample x 2 dim  (projection on 2D sub-space)
   """
    pass


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
    """Plot High dimemnsionnal State using TSNE method
   'euclidean, 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean, 'cosine, 'correlation, 'hamming, 'jaccard, 'chebyshev,
   'canberra, 'braycurtis, 'mahalanobis', VI=None) 'yule, 'matching, 'dice, 'kulsinski, 'rogerstanimoto, 'russellrao, 'sokalmichener, 'sokalsneath,

   Xtsne= da.plot_cluster_tsne(Xtrain_dist, Xcluster_label=None, perplexity=40, ncomponent=2, isprecompute=True)

   Xtrain_dist= sci.spatial.distance.squareform(sci.spatial.distance.pdist(Xtrain_d,
               metric='cityblock', p=2, w=None, V=None, VI=None))
   """
    pass


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
    pass


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
    pass


def plot_distribution_density(Xsample, kernel="gaussian", N=10, bandwith=1 / 10.0):
    pass

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
    pass


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

    pass


def plot_XY_plotly(xx, yy, towhere="url"):
    """ Create Interactive Plotly   """
    pass

    """
  trace = go.Scatter(x= xx, y= yy, marker= Marker(
            size=16,
            cmax=39,
            cmin=0,
            color=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            colorbar=ColorBar(title='Colorbar' )),  colorscale='Viridis')
  """

def plot_XY_seaborn(X, Y, Zcolor=None):
    pass


"""
def plot_cluster_embedding(Xmat, title=None):
   # Scale and visualize the embedding vectors
   x_min, x_max=np.min(Xmat, 0), np.max(Xmat, 0)
   Xmat=(Xmat - x_min) / (x_max - x_min)
   nX= Xmat.shape[0]

   plt.figure()
   ax=plt.subplot(111)
   colors= np.arange(0, nX, 5)
   for i in range(nX):
      plt.text(Xmat[i, 0], Xmat[i, 1], str(labels[i]), color=plt.cm.Set1(colors[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

   if hasattr(offsetbox, 'AnnotationBbox'):
      # only print thumbnails with matplotlib > 1.0
      shown_images=np.array([[1., 1.]])  # just something big
      for i in range(digits.data.shape[0]):
         dist=np.sum((Xmat[i] - shown_images) ** 2, 1)
         if np.min(dist) < 4e-3: continue  # don't show points that are too close

         shown_images=np.r_[shown_images, [Xmat[i]]]
         imagebox=offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), Xmat[i])
         ax.add_artist(imagebox)
   plt.xticks([]), plt.yticks([])
   if title is not None:  plt.title(title)
"""
