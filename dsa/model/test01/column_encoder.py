# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
import os
import sys
import warnings

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import (NMF, PCA, LatentDirichletAllocation,
                                   TruncatedSVD)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder,
                                   OneHotEncoder)

from sklearn.utils import check_random_state, murmurhash3_32




class MinHashEncoder(BaseEstimator, TransformerMixin):
    """
    minhash method applied to ngram decomposition of strings
    n_components : output dimension.

    """

    def __init__(self, n_components, ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.n_components = n_components

    def get_unique_ngrams(self, string, ngram_range):
        """
        Return a list of different n-grams in a string
        """
        spaces = " "  # * (n // 2 + n % 2)
        string = spaces + " ".join(string.lower().split()) + spaces
        ngram_list = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            string_list = [string[i:] for i in range(n)]
            ngram_list += list(set(zip(*string_list)))
        return ngram_list

    def minhash(self, string, n_components, ngram_range):
        min_hashes = np.ones(n_components) * np.infty
        grams = self.get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = self.get_unique_ngrams(" Na ", self.ngram_range)
        for gram in grams:
            hash_array = np.array(
                [murmurhash3_32("".join(gram), seed=d, positive=True)
                 for d in range(n_components)]
            )
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes / (2 ** 32 - 1)

    def fit(self, X, y=None):

        self.hash_dict = {}
        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.minhash(
                    x, n_components=self.n_components, ngram_range=self.ngram_range
                )
        return self

    def transform(self, X):

        X_out = np.zeros((len(X), self.n_components))

        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.minhash(
                    x, n_components=self.n_components, ngram_range=self.ngram_range
                )

        for i, x in enumerate(X):
            X_out[i, :] = self.hash_dict[x]

        return X_out

