'''
be sure to add your source directory - PyCharm: right click on the source root and click
Mark Directory as --> Sources Root

Code is on page 286 but refers to code from chatpers 3 and 4.
'''
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

'''
Unlike the book, I'm not fetching from the NLTK dataset. I have my own version,
which is a little cleaner than the one on NLTK.
'''

data_df= pd.read_csv('fetch_20newsgroups.csv')
print('Original shape:', data_df.shape)
print(data_df.head(10))

# data reprocessing and normalization
total_nulls = data_df[data_df.Article.str.strip() == ''].shape[0]
print('Empty documents:', total_nulls)

data_df = data_df[~(data_df.Article.str.strip() == '')]
print('New shape:', data_df.shape)

# starting on page 290 - follow my code! Author's code won't work.
import nltk
from chapter_3.assignment.text_normalizer import TextNormalizer # this will be your path

tn = TextNormalizer()
# normalize the corpus
import time
start = time.time()
norm_corpus = tn.normalize_corpus(corpus=data_df['Article'], html_stripping=True,
                                  contraction_expansion=True, accented_char_removal=True,
                                  text_lower_case=True, text_lemmatization=True,
                                  special_char_removal=True, remove_digits=True,
                                  stopword_removal=True)
full_time = round(time.time() - start, 2)
print('Normalizing finished in ', str(full_time))

data_df['Clean Article'] = norm_corpus
# view sample data
data_df = data_df[['Article', 'Clean Article', 'Target Label', 'Target Name']]
print(data_df.head())

data_df = data_df.dropna(axis=0, how='any').reset_index(drop=True)
print(data_df.info())

data_df.to_csv('clean_newsgroups.csv')