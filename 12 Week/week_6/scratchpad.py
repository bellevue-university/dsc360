from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_df_sample(subset):
    data = fetch_20newsgroups(subset=subset, shuffle=True, remove=('headers', 'footers', 'quotes'))
    data_labels_map = dict(enumerate(data.target_names))
    corpus, target_labels, target_names = (data.data, data.target,
                                           [data_labels_map[label] for label in data.target])

    data_df = pd.DataFrame({'Article': corpus, 'Target Label ': target_labels,
                            'Target Name': target_names})

    data_df_sample = data_df.sample(frac=0.5, replace=True, random_state=1)
    return data_df_sample
'''
test_df_sample = get_df_sample('test')
print(test_df_sample.shape)
train_df_sample = get_df_sample('train')
print(train_df_sample.shape)
'''
print(pd.DataFrame(
    [['NB', 0.712211, 0.726599],
    ['LR', 0.738673, 0.75303],
    ['SVM', 0.751011, 0.751011],
    ['SGD', 0.752177, 0.765657],
    ['RF', 0.526664, 0.544613],
    ['GBM', 0.53853, 0.547811]],
    columns=['Model', 'CV Score (TF-IDF)', 'Test Score (TF-IDF)']))
