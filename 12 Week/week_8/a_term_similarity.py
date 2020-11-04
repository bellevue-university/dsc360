# Analyzing Term Similarity - starting on page 459
import numpy as np
import pandas as pd
from scipy.stats import itemfreq

def vectorize_terms(terms):
    terms = [term.lower() for term in terms]
    terms = [np.array(list(term)) for term in terms]
    terms = [np.array([ord(char) for char in term]) 
                for term in terms]
    return terms

root = 'Believe'
term1 = 'believe'
term2 = 'bargain'
term3 = 'Elephant'

terms = [root, term1, term2, term3]
print('Terms:\n', terms, '\n')

# Character vectorization
term_vectors = vectorize_terms(terms)

# show vector representations
vec_df = pd.DataFrame(term_vectors, index=terms)
print('Term vectors:\n', vec_df, '\n')

root_term = root
other_terms = [term1, term2, term3]

root_term_vec = vec_df[vec_df.index == root_term].dropna(axis=1).values[0]
other_term_vecs = [vec_df[vec_df.index == term].dropna(axis=1).values[0]
                   for term in other_terms]

# Hamming Distance starting on page 461
print('Hamming Distance:')
def hamming_distance(u, v, norm=False):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return (u != v).sum() if not norm else (u != v).mean()

# compute Hamming distance
for term, term_vector in zip(other_terms, other_term_vecs):
    try:
        print('Hamming distance between root: {} and term: {} is {}'.
              format(root_term, term, hamming_distance(root_term_vec, term_vector,
                                                       norm=False)))
    except ValueError as ve:
        print('An error occurred:' + str(ve))
        continue

# computer normalized Hamming distance
for term, term_vector in zip(other_terms, other_term_vecs):
    try:
        print('Normalized Hamming distance between root: {} and term: {} is {}'
              .format(root_term, term,round(hamming_distance(root_term_vec,
                                                             term_vector, norm=True), 2)))
    except ValueError as ve:
        print('An error occurred:' + str(ve))
        continue

print('\nManhattan Distance:')
# Manhattan Distance starting on page 463
def manhattan_distance(u, v, norm=False):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return abs(u - v).sum() if not norm else abs(u - v).mean()

# compute Manhattan distance
for term, vector_term in zip(other_terms, other_term_vecs):
    try:
        print('Manhattan distance between root: {} and term: {} is {}'
              .format(root_term, term, manhattan_distance(root_term_vec,
                                                          term_vector, norm=False)))
    except ValueError as ve:
        print('An error occurred:' + str(ve))
        continue

# computer normalized Manhattan distance
for term, term_vector in zip(other_terms, other_term_vecs):
    try:
        print('Normalized Manhattan distance between root: {} and term: {} is {}'
              .format(root_term, term, round(manhattan_distance(root_term_vec, term_vector,
                                                                norm=True), 2)))
    except ValueError as ve:
        print('An error occurred:' + str(ve))
        continue

# Euclidean Distance starting on page 463
print('\nEuclidean Distance:')
def euclidean_distance(u,v):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    distance = np.sqrt(np.sum(np.square(u - v)))
    return distance

# compute Euclidean distance
for term, term_vector in zip(other_terms, other_term_vecs):
    try:
        print('Euclidean distance between root: {} and term: {} is {}'
              .format(root_term, term, round(euclidean_distance(root_term_vec,
                                                                term_vector), 2)))
    except ValueError as ve:
        print('An error occurred:' + str(ve))
        continue

# Levenshtein Edit Distance starting on page 467
print('\nLevenshtein Edit Distance')
import copy
def levenshtein_edit_distance(u, v):
    # convert to lower case
    u = u.lower()
    v = v.lower()
    # base cases
    if u == v: return 0
    elif len(u) == 0: return len(v)
    elif len(v) == 0: return len(u)
    # initialize edit distance matrix
    edit_matrix = []
    # initialize two distance matrices 
    du = [0] * (len(v) + 1)
    dv = [0] * (len(v) + 1)
    # du: the previous row of edit distances
    for i in range(len(du)):
        du[i] = i
    # dv : the current row of edit distances    
    for i in range(len(u)):
        dv[0] = i + 1
        # compute cost as per algorithm
        for j in range(len(v)):
            cost = 0 if u[i] == v[j] else 1
            dv[j + 1] = min(dv[j] + 1, du[j + 1] + 1, du[j] + cost)
        # assign dv to du for next iteration
        for j in range(len(du)):
            du[j] = dv[j]
        # copy dv to the edit matrix
        edit_matrix.append(copy.copy(dv))
    # compute the final edit distance and edit matrix    
    distance = dv[len(v)]
    edit_matrix = np.array(edit_matrix)
    edit_matrix = edit_matrix.T
    edit_matrix = edit_matrix[1:,]
    edit_matrix = pd.DataFrame(data=edit_matrix,
                               index=list(v),
                               columns=list(u))
    return distance, edit_matrix
'''
# Computer Levenshtein distance
for term, term_vector in zip(other_terms, other_term_vecs):
    edit_d, edit_m = levenshtein_edit_distance(root_term_vec, term_vector)
    print('Computing distance between root: {} and term: {}'.format(root_term,
                                                                    term))
    print('Levenshtein edit distance is {}'.format(edit_d))
    print('The complete edit distance matrix is depicted below')
    print(edit_m)
    print('-'*30)

print('\n')
'''

# Cosine Distance and Similarity starting on page 473
print('\nCosine Distance:')
def boc_term_vectors(word_list):
    word_list = [word.lower() for word in word_list]
    unique_chars = np.unique(np.hstack([list(word)
                                        for word in word_list]))
    word_list_term_counts = [{char: count for char, count in np.stack(
        np.unique(list(word), return_counts=True), axis=1)} for word in word_list]

    boc_vectors = [np.array([int(word_term_counts.get(char, 0))
                             for char in unique_chars])
                   for word_term_counts in word_list_term_counts]
    return list(unique_chars), boc_vectors

feature_names, feature_vectors = boc_term_vectors(terms)
boc_df = pd.DataFrame(feature_vectors, columns=feature_names, index=terms)
print('Bag of characters vectors:\n', boc_df)

def cosine_distance(u, v):
    distance = 1.0 - (np.dot(u, v) / 
                        (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.square(v)))))
    return distance

root_term_boc = boc_df[vec_df.index == root_term].values[0]
other_term_bocs = [boc_df[vec_df.index == term].values[0]
                   for term in other_terms]

# Compute Cosine Distance
print('\nCompute Cosine Distance:')
for term, boc_term in zip(other_terms, other_term_bocs):
    print('Analyzing similarity between root: {} and term: {}'.format(root_term, term))
    distance = round(cosine_distance(root_term_boc, boc_term), 2)
    similarity = 1 - distance                                                           
    print('Cosine distance  is {}'.format(distance))
    print('Cosine similarity  is {}'.format(similarity))
    print('-'*40)

