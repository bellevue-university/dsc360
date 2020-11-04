import numpy as np
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

# THIS FILE TAKES A WHILE TO RUN

# set the data as in the last two files
data_df = pd.read_csv('week_6/clean_newsgroups.csv')
# just in case there are blanks.
data_df['Clean Article'].dropna()
train_corpus, test_corpus, train_label_nums, test_label_nums, \
    train_label_names, test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                                          np.array(data_df['Target Label']),
                                                          np.array(data_df['Target Name']),
                                                          test_size=0.33, random_state=42)

# this is not in the book but is part of the author's normalization.py file
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

# starting on page 323
# Word2Vec Embeddings with Classification Models
def document_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype='float64')
        num_words = 0. 

        for word in words:
            if word in vocabulary:
                num_words = num_words + 1. 
                feature_vector = np.add(feature_vector, model.wv[word])
        if num_words:
            feature_vector = np.divide(feature_vector, num_words)
        
        return feature_vector
    
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)

# tokenize corpus
tokenized_train = [tokenize_text(text) for text in train_corpus]
tokenized_test = [tokenize_text(text) for text in test_corpus]

# generate word2vec word embeddings
import gensim
# build word2vec model
w2v_num_features = 1000
w2v_model = gensim.models.Word2Vec(tokenized_train, size=w2v_num_features, window=100, 
                                   min_count=2, sample=1e-3, sg=1, iter=5, workers=10)

'''
generate document level embeddings
remember we only use train dataset vocabulary embeddings
so that test dateset truly remains an unseen dataset
'''
# generate averaged word vector features from word2vec model
avg_wv_train_features = document_vectorizer(corpus=tokenized_train, model=w2v_model, 
                                            num_features=w2v_num_features)
avg_wv_test_features = document_vectorizer(corpus=tokenized_test, model=w2v_model, 
                                           num_features=w2v_num_features)

print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape,
      'Test features shape:', avg_wv_test_features.shape, '\n')

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

svm = SGDClassifier(loss='hinge', penalty='l2', max_iter=50, tol=1e-3)
svm.fit(avg_wv_train_features, train_label_names)
svm_w2v_cv_scores = cross_val_score(svm, avg_wv_train_features, train_label_names, cv=5)
svm_w2v_cv_mean_score = np.mean(svm_w2v_cv_scores)
print('CV Accuracy (5-fold):', svm_w2v_cv_scores)
print('Mean CV Accuracy:', svm_w2v_cv_mean_score)
svm_w2v_test_score = svm.score(avg_wv_test_features, test_label_names)
print('Test Accuracy:', svm_w2v_test_score, '\n')

# skipping GloVe and FastText - same ideas as W2V and take a while to run

# skipping the neural network - also computationally expensive and hasn't been introduced
