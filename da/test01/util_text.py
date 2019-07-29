# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:14:42 2019

@author: zenbook
"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import *
# make the ROC curve plots from both models
from sklearn import metrics
# split the data into training and test sets
# we pass the predictive features with no "red"
# as the predictor data
from sklearn.cross_validation import train_test_split
# Prepare and run Random Forest
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
# define the count vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

stemmer = PorterStemmer()




# Function to convert a raw text to a string of words
# The input is a single string (a raw text), and 
# the output is a single string (a preprocessed text)

def text_to_words(raw_text):
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 3. Remove Stopwords. In Python, searching a set is much faster than 
    # searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]  
    # 5. Stem words. Need to define porter stemmer above
    singles = [stemmer.stem(word) for word in meaningful_words]
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( singles ))  




# what does the function do? 

# fifth text, unprocessed
print(wine_data.Winemakers_Notes[5])
# fifth text, processed
print(text_to_words(wine_data.Winemakers_Notes[5]))



# apply the function to the winemaker's notes text column
# using list comprehension 
processed_texts = [text_to_words(text) for text in wine_data.Winemakers_Notes]




# For predictive modeling, we would like to encode the top words as 
# features. We select words that occur in at least 50 documents.
# We use the "binary = True" option to create dichotomous indicators.
# If a given text contained a word, the variable is set to 1, otherwise 0
# The ngram_range specifies that we want to extract single words
# other possibilities include bigrams, trigrams, etc.

vectorizer_count = CountVectorizer(ngram_range=(1, 1), 
                                   min_df = 50, binary = True)

# and apply it to the processed texts
pred_feat_unigrams = vectorizer_count.fit_transform(processed_texts)









# turn the dtm matrix to a numpy array to sum the columns
pred_feat_array = pred_feat_unigrams.toarray()
# Sum up the counts of each word
dist = np.sum(pred_feat_array, axis=0)
# extract the names of the features
vocab = vectorizer_count.get_feature_names() 
# make it a dataframe
topwords = pd.DataFrame(dist, vocab, columns = ["Word_in_Num_Documents"])
# add 'word' as a column
topwords = topwords.reset_index()
# sort the words by document frequency
topwords = topwords.sort_values(by = 'Word_in_Num_Documents', 
                                ascending=False)





# plot the top 15 words

# set the palette for 15 colors
current_palette = sns.cubehelix_palette(15, start = .3, reverse=True)
sns.set_palette(current_palette, n_colors = 15)

# plot the top words and set the axis labels
topwords_plot = sns.barplot(x = 'index', y="Word_in_Num_Documents", 
             data=topwords[0:15])
topwords_plot.set(xlabel='Word')
topwords_plot.set(ylabel='Number of Documents')







# train test split. using the same random state as before ensures
# the same observations are placed into train and test set as in the
# previous posts

X_train, X_test, y_train, y_test = train_test_split(pred_feat_unigrams, 
                            wine_data["Varietal_WineType_Name"], 
                            test_size=0.30, random_state=42)






# Create the random forest object
rforest = RandomForestClassifier(n_estimators = 500, n_jobs=-1)
# Fit the training data 
rforest_model = rforest.fit(X_train,y_train)
### Do prediction (class) on test data
preds_class_rf = rforest_model.predict(X_test)
# confusion matrix: how well do our predictions match the actual data?
pd.crosstab(pd.get_dummies(y_test)['Red Wines'],preds_class_rf)









# plot the top features

# set the palette for 25 colors
# the _r gives a reverse ordering
# to keep the darker colors on top
sns.set_palette("Blues_r", n_colors = 25)

# extract the top features
feature_names = np.array(vectorizer_count.get_feature_names())  
df_featimport = pd.DataFrame([i for i in zip(feature_names,
                rforest_model.feature_importances_)], 
                columns=["features","importance"])

# plot the top 25 features
top_features = sns.barplot(x="importance", y="features", 
             data=df_featimport.sort('importance', ascending=False)[0:25])
top_features.set(xlabel='Feature Importance')
top_features.set(ylabel='Feature')




# redefine the predictor matrix without the word "red"
# using pred_feat_array created above, turn into dataframe
pred_feat_nored = pd.DataFrame(pred_feat_array,
                         columns = vectorizer_count.get_feature_names())
# and drop the column with the indicator for the word "red"
pred_feat_nored = pred_feat_nored.drop('red', axis = 1)



X_train_nr, X_test_nr, y_train_nr, y_test_nr = train_test_split(pred_feat_nored, 
                            wine_data["Varietal_WineType_Name"], 
                            test_size=0.30, random_state=42)

# define the model. same hyper-parameters as above 
rforest_nr = RandomForestClassifier(n_estimators = 500, n_jobs=-1)
# Fit the training data 
rforest_model_nr = rforest_nr.fit(X_train_nr,y_train_nr)
### Do prediction (class) on test data
preds_class_rf_nr = rforest_model_nr.predict(X_test_nr)
# confusion matrix: how well do our predictions match the actual data?
pd.crosstab(pd.get_dummies(y_test_nr)['Red Wines'],preds_class_rf_nr
            
            
            
            
            
            
            
            
            
# compute the probability predictions from both models
# model with all unigrams
preds_rf = rforest_model.predict_proba(X_test)
# model with "red" removed from predictor matrix
preds_rf_nr = rforest_model_nr.predict_proba(X_test_nr)


plt.figure(figsize=(10,10))
fpr, tpr, _ = metrics.roc_curve(pd.get_dummies(y_test)['Red Wines'], 
                                preds_rf[:,0])
auc1 = metrics.auc(fpr,tpr) 

plt.plot(fpr, tpr,label='AUC Predictors All Unigrams: %0.3f' % auc1,
         color='red', linewidth=2)

fpr, tpr, _ = metrics.roc_curve(pd.get_dummies(y_test_nr)['Red Wines'], 
                                preds_rf_nr[:,0])
auc1 = metrics.auc(fpr,tpr) 

plt.plot(fpr, tpr,label="AUC Predictors Without 'Red': %0.3f" % auc1,
         color='blue', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1) 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate') 
plt.title('ROC') 
plt.grid(True)
plt.legend(loc="lower right")




# insight into top predictor: plot presence of 
# 'tannin' in red vs. white wine texts

# merge wine type and feature information
winetype_features = pd.concat([wine_data["Varietal_WineType_Name"],
                 pd.DataFrame(pred_feat_unigrams.toarray(), 
                 columns = vectorizer_count.get_feature_names())], axis = 1)

# define the colors to use in the plot                  
flatui = ['darkred','orange' ]
sns.set_palette(sns.color_palette(flatui))

# plot the percentage of texts with the word 'tannin' in them
# for red and white wines, respectively
tannin_plot = sns.barplot(x = "Varietal_WineType_Name", y= "tannin", 
                          data=winetype_features, capsize=.2);
tannin_plot.set(xlabel='Wine Type', 
                ylabel = "Percentage of Texts With the Word 'Tannin'")








# Creating Unigram & Bigram Vectors


vectorizer =TfidfVectorizer(ngram_range=(1,2))

X_ngrams=vectorizer.fit_transform(processed)

The term frequency (tf) measures the occurrenc









# Creating Unigram & Bigram Vectors


vectorizer =TfidfVectorizer(ngram_range=(1,2))

X_ngrams=vectorizer.fit_transform(processed)
            
            
            
 #Train/Test Split


X_train,X_test,y_train,y_test=train_test_split(X_ngrams,y,test_size=0.2,stratify=y)

 

# Running the Classsifier


clf=LogisticRegression()

clf.fit(X_train,y_train)           
