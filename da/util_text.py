# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas


#####################################################################################################
######### Term Frequency   ##########################################################################
If you need the term frequency (term count) vectors for different tasks, use Tfidftransformer.
If you need to compute tf-idf scores on documents within your “training” dataset, use Tfidfvectorizer
If you need to compute tf-idf scores on documents outside your “training” dataset, use either one, both will work.



#####################################################################################################
### The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.

feature_extraction.text.CountVectorizer([ÿ])  Convert a collection of text documents to a matrix of token counts
feature_extraction.text.HashingVectorizer([ÿ])  Convert a collection of text documents to a matrix of token occurrences
feature_extraction.text.TfidfVectorizer([ÿ])  Convert a collection of raw documents to a matrix of TF-IDF features.



"""
import copy
import os
import re
import math
from collections import Counter
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sci


import sklearn as sk
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer



import nltk
from nltk.corpus import stopwords

# Stemming and Lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


try :
    import spacy
    import gensim
except Exception as e :
    print(e)


print("os.getcwd", os.getcwd())

dd = [ 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



#################################################################################
#################################################################################
porter = PorterStemmer()
def coltext_stemporter(text):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split(" ")
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


wordnet = WordNetLemmatizer()
def coltext_lemmatizer( text ):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split()
    stemmed_tokens = [ wordnet.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


snowball = SnowballStemmer('english')
def coltext_stemmer( text ):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split()
    stemmed_tokens = [ snowball.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


stopwords_dict_en =  stopwords.words('english')
def coltext_stopwords( text ,  stopwords_dict=stopwords_dict_en):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split()
    stemmed_tokens = [ token for token in tokens if token not in stopwords_dict]
    return ' '.join(stemmed_tokens)




def pd_coltext_extract_tag(df, col_tagtoextract, coltext):
    '''
    Function that one hot encodes a feature from a comment column
    Arguments:
        df:           dataframe
        col_tagtoextract:      column to create
        coltext:  column where the comments are
    Returns:
        df:           dataframe with new column
    '''
    df[col_tagtoextract] = df[coltext].str.contains(col_tagtoextract)

    #One hot encode that feature:
    # df = pd_col_to_onehot(df, [col_tagtoextract])
    return df


from collections import Counter

def pd_coltext_wordfreq_df(df, coltext, nb_to_show=20):
    """
    :param df:
    :param coltext:  text where word frequency should be extracted
    :param nb_to_show:
    :return:
    """
    results = Counter()
    df[coltext].str.strip('{}') \
            .str.replace('"', '') \
            .str.lstrip('\"') \
            .str.rstrip('\"') \
            .str.split(',') \
            .apply(results.update)

    ll = {"text": [], "freq": []}
    for amenity in results.most_common(nb_to_show):
        ll["text"].append(amenity[0])
        ll["freq"].append(amenity[1])
    ll = pd.DataFrame(ll)
    return ll



def pd_coltext_wordfreq(df, coltext) :
  """
  Word Frequency
  :param df:
  :param coltext:
  :return: ddict, df
  """
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer()
  X=cv.fit_transform( df[coltext] )
  word_list = cv.get_feature_names()
  count_list = X.toarray().sum(axis=0)
  ddict = {word_list[i]  : count_list[i]  for i in range(len(word_list)) }
  df = pd_fromdict(ddict, [ "name", "freq" ] )
  return ddict, df




def pd_fromdict(ddict, colname ) :
    """
    :param ddict:
    :param colname:
    :return:
    """
    colname = ("c0", "c1" ) if colname is None else colname
    klist, xlist = [] , []
    for k,x in ddict.items():
        klist.append(k)
        xlist.append(x)
    df = pd.DataFrame( {colname[0] : klist, colname[1] : xlist }  )
    df = df.sort_values(by=colname[1], ascending=False)
    return df



def pd_coltext_tfidf(df, coltext, word_tokeep=None, word_minfreq=1):
    '''
    Function that adds tf-idf of a given column for words in a text corpus.
    Arguments:
        df:             original dataframe
        word_tokeep: corpus of words to look into
        col_tofilter:   column of df to apply tf-idf to
    Returns:
        concat_df:      dataframe with a new column for each word
      https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    '''
    from sklearn.feature_extraction.text import CountVectorizer

    if word_tokeep is None :
      cv = CountVectorizer()
      X = cv.fit_transform( df[coltext])
      word_tokeep = cv.get_feature_names()
      count_list = np.asarray(X.sum(axis=0))
      word_dict = dict(zip(word_tokeep, count_list))
      print(len(word_tokeep))


    vectorizer = TfidfVectorizer()
    vectorizer.fit(word_tokeep)
    v = vectorizer.transform(df[coltext])
    v =  v.toarray()
    print(v.shape)

    df_vector = pd.DataFrame(v, columns=word_tokeep)
    # df_new = pd.concat([df, df_vector],axis=1)
    return df_vector


def pd_coltext_hashing(df, coltext, n_features=20):
    '''
    Function that adds Hash a given column for words in a text corpus.
    Arguments:
        df:             original dataframe
        word_tokeep: corpus of words to look into
        col_tofilter:   column of df to apply tf-idf to

    Returns:
        concat_df:      dataframe with a new column for each word
    '''
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(n_features=n_features)
    vector = vectorizer.transform(df[coltext])
    print(vector.shape)
    colname =  [ "c" + str(i) for i  in range(0, n_features) ]

    df_vector = pd.DataFrame(vector.toarray(), columns= colname )
    return df_vector












