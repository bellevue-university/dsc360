# starting on page 203
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None, 'display.max_columns', None)
# the next line is only for Jupyter Notebook
#%matplotlib inline

# building a corpus of documents
corpus = [
    'The sky is blue and beautiful.',
    'Love this blue and beautiful sky!',
    'The quick brown fox jumps over the lazy dog.',
    'A king\'s breakfast has sausages, ham, bacon, eggs, toast, and beans.',
    'I love green eggs, ham, saussages, and bacon!',
    'The brown fox is quick and the blue dog is lazy!',
    'The sky is very blue and the sky is very beautiful today.',
    'The dog is lazy but the brown fox is quick!'
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals',
          'weather', 'animals']

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
print(corpus_df, '\n')

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('English')

def normalize_document(doc):
    # lowercase and remove special characters\whitespace
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    #tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(corpus)
print(norm_corpus, '\n')

# starting on page 208
print('Bag of Words Model')
# starting on page 208
from sklearn.feature_extraction.text import CountVectorizer
# get bag of words features in sparse format
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
# view non-zero feature positions in the sparse matrix
print(cv_matrix, '\n')

# view dense representation
# warning - might give a memory error if the data is too big
cv_matrix = cv_matrix.toarray()
print(cv_matrix, '\n')

# get all unique words in the corpus
vocab = cv.get_feature_names()
#show document feature vectors
cv_df = pd.DataFrame(cv_matrix, columns=vocab)
print(cv_df, '\n')

# you can set the n-gram range to 1,2 to get unigrams as well as bigrams
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_corpus)
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
bv_df = pd.DataFrame(bv_matrix, columns=vocab)
print(bv_df, '\n')

# starting on page 213
print('tfidf transformer:')
from sklearn.feature_extraction.text import TfidfTransformer
tt = TfidfTransformer(norm = 'l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix)
tt_matrix = tt_matrix.toarray()
vocab = cv.get_feature_names()
print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab), '\n')

# tfidfvectorizer, page 214
print('tfidf vectorizer:')
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()
# this part is not in the book - save the tv_matrix for use in the next file (b_document_similiarity.py)
import os
np.save(os.getcwd() + '\\week_5\\data\\tv_matrix.npy', tv_matrix)

vocab = tv.get_feature_names()
print(pd.DataFrame(np.round(tv_matrix, 2), columns=vocab), '\n')

# Understanding the TF-DF Model - starting on page 215
# get unique words as feature names
# different output than book
unique_words = list(set([word for doc in [doc.split() for doc in norm_corpus] for word in doc]))
def_feature_dict = {w: 0 for w in unique_words}
print('Feature Names:', unique_words)
print('Default Feature Dict:', def_feature_dict, '\n')

# page 216
from collections import Counter
# build bag of words features for each document - term frequencies
bow_features = []
for doc in norm_corpus:
    bow_feature_doc = Counter(doc.split())
    all_features = Counter(def_feature_dict)
    bow_feature_doc.update(all_features)
    bow_features.append(bow_feature_doc)

bow_features = pd.DataFrame(bow_features)
print('BOW features:\n', bow_features)

# DF - starting on page 216
import scipy.sparse as sp 
feature_names = list(bow_features.columns)

# build the document frequency matrix
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df # adding 1 to smoothen idf later

# show smoothened document frequencies
print('Smooth DF:\n', pd.DataFrame([df], columns=feature_names), '\n')

# IDF - page 217
# compute inverse document frequencies
total_docs = 1 + len(norm_corpus)
idf = 1.0 + np.log(float(total_docs) / df) 

# show smoothened IDFs
print('Smooth IDFs:\n', pd.DataFrame([np.round(idf, 2)], columns=feature_names), '\n')

# compute idf diagonal matrix
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf_dense = idf_diag.todense()

# print the idf diagonal matrix
print('Diagonal matrix:\n', pd.DataFrame(np.round(idf_dense, 2)), '\n')

# compute tfidf feature matrix - page 218
tf = np.array(bow_features, dtype='float64')
tfidf = tf * idf
# view raw tfidf feature matrix
print('Raw TF-IDF feature matrix\n', pd.DataFrame(np.round(tfidf, 2), columns=feature_names), '\n')

# computer l2 norms
from numpy.linalg import norm
norms = norm(tfidf, axis=1)

# print norms for each document
print('Norms:\n', np.round(norms, 3), '\n')

# compute normalized tfidf
norm_tfidf = tfidf / norms[:, None]

# show final tfidf feature matrix
print('Final TF-DF feature matrix:\n',  pd.DataFrame(np.round(norm_tfidf, 2), columns=feature_names), '\n')

# Extracting Features for New Documents - page 220
new_doc = 'the sky is green today'
print('New doc features:\n', pd.DataFrame(np.round(tv.transform([new_doc]).toarray(), 2), columns=tv.get_feature_names()), '\n')
