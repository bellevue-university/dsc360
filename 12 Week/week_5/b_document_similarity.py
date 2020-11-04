import os
import numpy as np
import pandas as pd

# load the tv_matrix save in the last file
tv_matrix = np.load(os.getcwd() + '\\week_5\\data\\tv_matrix.npy')

# Document Similarity - staring on page 221
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
print('Similarity matrix DF:\n', similarity_df, '\n')