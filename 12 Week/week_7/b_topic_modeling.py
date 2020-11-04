import os
import numpy as np
import pandas as pd

# page 365
DATA_PATH = 'nipstxt/'
print(os.listdir(DATA_PATH))

# page 366
folders = ['nips{0:02}'.format(i) for i in range(0, 13)]
# Read all texts into a list.
papers = []
for folder in folders:
    file_names = os.listdir(DATA_PATH + folder)
    for file_name in file_names:
        with open(DATA_PATH + folder + '/' + file_name, encoding='utf-8',
                  errors='ignore', mode='r+') as f:
            data = f.read()
        papers.append(data)
# save the papers list, you'll need this a bit later on

print('Length of papers:\n', len(papers), '\n')

print('Paper fragment:\n', papers[0][:1000], '\n')

# Text Wrangling - starting on page 367
import nltk

stop_words = nltk.corpus.stopwords.words('english')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

def normalize_corpus(papers):
    norm_papers = []
    for paper in papers:
        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens
                        if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))
        if paper_tokens:
            norm_papers.append(paper_tokens)
    return norm_papers

norm_papers = normalize_corpus(papers)
print('Length of normalized papers:\n', len(norm_papers), '\n')

# Text Representation with Feature Engineering
# page 369
import gensim

bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20, delimiter=b'_')
bigram_model = gensim.models.phrases.Phraser(bigram)

# sample demonstration
print('Bigram model: \n', bigram_model[norm_papers[0]][:50], '\n')

# page 370
norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Sample word to number mappings:\n', list(dictionary.items())[:15], '\n')
print('Total vocabulary size: ', len(dictionary), '\n')

# Filter out words that occur in fewer than 20 documents, or more than 50%
# of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total vocabulary size:\n', len(dictionary), '\n')

# Transforming corpus into bag of words vectors
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print('Bag of words:\n', bow_corpus[1][:50], '\n')

# viewing actual terms and their counts
print('Terms and counts:\n', [(dictionary[idx], freq) for idx, freq in bow_corpus[1][:50]], '\n')

# total papers in the corpus
print('Total number of papers:\n', len(bow_corpus), '\n')

# Latent Semantic Indexing - page 372
TOTAL_TOPICS = 10
from gensim.models import LsiModel
lsi_bow = LsiModel(bow_corpus, id2word=dictionary, num_topics=TOTAL_TOPICS,
                   onepass=True, chunksize=1740, power_iters=1000)

for topic_id, topic in lsi_bow.print_topics(num_topics=10, num_words=20):
    print('Topic #' + str(topic_id+1)+':')
    print(topic, '\n')

for n in range(TOTAL_TOPICS):
    print('Topic #' + str(n+1)+':')
    print('='*50)
    d1 = []
    d2 = []
    for term, wt in lsi_bow.show_topic(n, topn=20):
        if wt >= 0:
            d1.append((term, round(wt, 3)))
        else:
            d2.append((term, round(wt, 3)))

    print('Direction 1:', d1)
    print('-'*50)
    print('Direction 2:', d2)
    print('-'*50, '\n')

# page 379
term_topic = lsi_bow.projection.u
singular_values = lsi_bow.projection.s
topic_document = (gensim.matutils.corpus2dense(lsi_bow[bow_corpus],
                                               len(singular_values)).T / singular_values).T
print(term_topic.shape, singular_values.shape, topic_document.shape)

document_topics = pd.DataFrame(np.round(topic_document.T, 3),
                               columns=['T' + str(i) for i in range(1, TOTAL_TOPICS+1)])
print(document_topics.head(5))

# page 380
document_numbers = [13, 250, 500]

for document_number in document_numbers:
    top_topics = list(document_topics.columns[np.argsort(
        -np.absolute(document_topics.iloc[document_number].values))[:3]])
    print('Document #' + str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print('Paper Summary:')
    print(papers[document_number][:500], '\n')

# Implementing LIS Topic Models from Scratch - starting on page 382
td_matrix = gensim.matutils.corpus2dense(corpus=bow_corpus,
                                         num_terms=len(dictionary))
print(td_matrix.shape)
print(td_matrix, '\n')

vocabulary = np.array(list(dictionary.values()))
print('Total vocabulary size:', len(vocabulary))
print(vocabulary, '\n')

from scipy.sparse.linalg import svds

u, s, vt = svds(td_matrix, k=TOTAL_TOPICS, maxiter=10000)
term_topic = u
topic_document = vt
print(term_topic.shape, singular_values.shape, topic_document.shape, '\n')

tt_weights = term_topic.transpose() * singular_values[:, None]
print(tt_weights.shape, '\n')

top_terms = 20
topic_key_term_idxs = np.argsort(-np.absolute(tt_weights), axis=1)[:, :top_terms]
topic_keyterm_weights = np.array([tt_weights[row, columns]
                                  for row, columns in list(zip(np.arange(TOTAL_TOPICS),
                                                               topic_key_term_idxs))])
topic_keyterms = vocabulary[topic_key_term_idxs]
topic_keyterms_weights = list(zip(topic_keyterms, topic_keyterm_weights))
for n in range(TOTAL_TOPICS):
    print('Topic #' + str(n+1) + ':')
    print('=' * 50)
    d1 = []
    d2 = []
    terms, weights = topic_keyterms_weights[n]
    term_weights = sorted([(t, w)
                           for t, w in zip(terms, weights)],
                          key = lambda row: -abs(row[1]))
    for term, wt in term_weights:
        if wt >= 0:
            d1.append((term, round(wt, 3)))
        else:
            d2.append((term, round(wt, 3)))

    print('Direction 1:', d1)
    print('-' * 50)
    print('Direction 2:', d2)
    print('-' * 50, '\n')

# page 387
document_topics = pd.DataFrame(np.round(topic_document.T, 3),
                               columns=['T' + str(i) for i in
                                        range(1, TOTAL_TOPICS+1)])
document_numbers = [13, 250, 500]

for document_number in document_numbers:
    top_topics = list(document_topics.columns[np.argsort(
        -np.absolute(document_topics.iloc[document_number].values))[:3]])

    print('Document #' + str(document_number) + ':')
    print('Dominant Topics (top 3):', top_topics)
    print('Paper Summary:')
    print(papers[document_number][:500], '\n')

# save data for use in the next file
import pickle
with open('data/papers.txt', 'wb') as fp:
    pickle.dump(papers, fp)

with open('data/bigram_model.txt', 'wb') as fp:
    pickle.dump(bigram_model, fp)

with open('data/norm_corpus_bigrams.txt', 'wb') as fp:
    pickle.dump(norm_corpus_bigrams, fp)

with open('data/bow_corpus.txt', 'wb') as fp:
    pickle.dump(bow_corpus, fp)

with open('data/dictionary.txt', 'wb') as fp:
    pickle.dump(dictionary, fp)
