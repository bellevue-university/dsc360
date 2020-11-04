# starts on page 292 - building train and test datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# load the data save to .csv from a_data.py
data_df = pd.read_csv('clean_newsgroups.csv')

# page 292
train_corpus, test_corpus, train_label_nums, test_label_nums, \
    train_label_names, test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                                          np.array(data_df['Target Label']),
                                                          np.array(data_df['Target Name']),
                                                          test_size=0.33, random_state=42)

from collections import Counter
trd  = dict(Counter(train_label_names))
tsd = dict(Counter(test_label_names))

print((pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
              columns=['Target Label', 'Train Count',
                       'Test Count']).sort_values(by=['Train Count', 'Test Count'], ascending=False)), '\n')

# starting on page 310 - note you haven't seen the breast cancer data set yet.
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=1234)

# train and build the model
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=5000)
logistic.fit(X_train, y_train)
# predict on test data and view confusion matrix
y_pred = logistic.predict(X_test)

# note this is a standard package, not the one in the book on page 310
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('Confusion matrix: \n', confusion_matrix)

# Performance Metrics, starting on page 312
positive_class = 1
TP = confusion_matrix[1, 1]
FP = confusion_matrix[0, 1]
TN = confusion_matrix[0, 0]
FN = confusion_matrix[1, 0]
print(TP, FP, TN, FN, '\n')

# accuracy
from sklearn.metrics import accuracy_score # standard
print('Framework Accuracy:', round(accuracy_score(y_test, y_pred), 5))
mc_acc = round((TP + TN) / (TP + TN + FP + FN), 5)
print('Manually Computed Accuracy:', mc_acc, '\n')

# precision
from sklearn.metrics import precision_score
print('Framework Precision:', round(precision_score(y_test, y_pred), 5))
mc_prec = round((TP) / (TP + FP), 5)
print('Manually Computed Precision:', mc_prec, '\n')

# recall
from sklearn.metrics import recall_score
print('Framework Recall:', round(recall_score(y_test, y_pred), 5))
mc_rec = round((TP) / (TP + FN), 5)
print('Manually computed Recall:', mc_rec, '\n')

from sklearn.metrics import f1_score
print('Framework F1-Score:', round(f1_score(y_test, y_pred), 5))
mc_f1 = round((2*mc_prec*mc_rec) / (mc_prec+mc_rec), 5)
print('Manually Computed F1-Score:', mc_f1)