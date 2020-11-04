import numpy as np
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

# set the data as in the last three files
data_df = pd.read_csv('week_6/clean_newsgroups.csv')
# just in case there are blanks.
data_df['Clean Article'].dropna()
train_corpus, test_corpus, train_label_nums, test_label_nums, \
    train_label_names, test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                                          np.array(data_df['Target Label']),
                                                          np.array(data_df['Target Name']),
                                                          test_size=0.33, random_state=42)

# Modeling Tuning - starting on page 329
# Tuning our Multinomial Naive Bayes Model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

mnb_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('mnb', MultinomialNB())])
param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'mnb__alpha': [1e-5, 1e-4, 1e-2, 1e-1, 1]}

gs_mnb = GridSearchCV(mnb_pipeline, param_grid, cv=5, verbose=2)
gs_mnb = gs_mnb.fit(train_corpus, train_label_names)

print(gs_mnb.best_estimator_.get_params(), '\n')

cv_results = gs_mnb.cv_results_
results_df = pd.DataFrame({'rank': cv_results['rank_test_score'],
                            'params': cv_results['params'], 
                            'cv score (mean)': cv_results['mean_test_score'],
                            'cv score (std)': cv_results['std_test_score']})
results_df = results_df.sort_values(by=['rank'], ascending=True)
pd.set_option('display.max_colwidth', 100)
print('Modeling tuning results DF:', results_df, '\n')

best_mnb_test_score = gs_mnb.score(test_corpus, test_label_names)
print('Test Accuracy:', best_mnb_test_score, '\n')

# Tuning our Logistic Regression Model 
from sklearn.linear_model import LogisticRegression
lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(penalty='l2', 
                          max_iter=100, random_state=42))])
param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'lr__C': [1, 5, 10]}

gl_lr = GridSearchCV(lr_pipeline, param_grid, cv=5, verbose=2)
gl_lr = gl_lr.fit(train_corpus, train_label_names)

# evaluate best tuned model on the test dataset
best_lr_test_score = gl_lr.score(test_corpus, test_label_names)
print('Test Accuracy:', best_lr_test_score, '\n')

# Tuning the Linear SVM model
from sklearn.svm import LinearSVC
svm_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('svm', LinearSVC(random_state=42))])
param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'svm__C': [0.01, 0.1, 1, 5]}

gs_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, verbose=2)
gs_svm = gs_svm.fit(train_corpus, train_label_names)

# evaluating best tuned model on the data set
best_svm_test_score = gs_svm.score(test_corpus, test_label_names)
print('Test Accuracy:', best_svm_test_score)

mnb_predictions = gs_mnb.predict(test_corpus)
unique_classes = list(set(test_label_names))
print('Unique classes:', unique_classes, '\n')