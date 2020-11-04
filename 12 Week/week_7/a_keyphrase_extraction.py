from nltk.corpus import gutenberg
from week_4.assignment.text_normalizer import TextNormalizer
import nltk
import pandas as pd
from operator import itemgetter

# page 352 (need to instantiate the class from chapter 3
tn = TextNormalizer()

# my code is a bit different than the author's but works with our
# TextNormalizer.
alice_txt = gutenberg.sents(fileids='carroll-alice.txt')
alice_list = list([' '.join(ts) for ts in alice_txt])
alice = pd.Series(alice_list)
norm_alice = tn.normalize_corpus(corpus=alice, text_lemmatization=False)

# print first line
print('\nAlice - before and after')
print(alice[0], '\n', norm_alice[0], '\n')

# page 353
def flatten_corpus(corpus):
    return ' '.join([document.strip()
                     for document in corpus])

# page 352
def compute_ngrams(sequence, n):
    return zip(*[sequence[index:]
                 for index in range(n)])

# page 353
def get_top_ngrams(corpus, ngram_val=1, limit=5):

    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(),
                              key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq)
                     for text, freq in sorted_ngrams]

    return sorted_ngrams

# page 353
print('Bigrams:\n', get_top_ngrams(corpus=norm_alice, ngram_val=2, limit=10), '\n')

# page 354
print('Trigrams:\n', get_top_ngrams(corpus=norm_alice, ngram_val=3, limit=10))

# page 355
print('Collocation Finder:\n')
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures # updated package

finder = BigramCollocationFinder.from_documents([item.split()
                                                for item
                                                in norm_alice])
bigram_measures = BigramAssocMeasures()
print('Bigram Association Measures:')
print(finder.nbest(bigram_measures.raw_freq, 10))
print(finder.nbest(bigram_measures.pmi, 10), '\n')

# page 356
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures # updated package

finder = TrigramCollocationFinder.from_documents([item.split()
                                                for item
                                                in norm_alice])
trigram_measures = TrigramAssocMeasures()
print('Trigram Association Measures:')
print(finder.nbest(trigram_measures.raw_freq, 10))
print(finder.nbest(trigram_measures.pmi, 10), '\n')

# page 357
sentences = """
Elephants are large mammals of the family Elephantidae 
and the order Proboscidea. Two species are traditionally recognised, 
the African elephant and the Asian elephant. Elephants are scattered 
throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male 
African elephants are the largest extant terrestrial animals. All 
elephants have a long trunk used for many purposes, 
particularly breathing, lifting water and grasping objects. Their 
incisors grow into tusks, which can serve as weapons and as tools 
for moving objects and digging. Elephants' large ear flaps help 
to control their body temperature. Their pillar-like legs can 
carry their great weight. African elephants have larger ears 
and concave backs while Asian elephants have smaller ears 
and convex or level backs.  
"""
sent_tokens = nltk.sent_tokenize(sentences)
print('Sentence tokenization:')
print(len(sent_tokens))
print(sent_tokens[:3], '\n')

print('Normalize text:')
sentences_series = pd.Series(sent_tokens)
norm_sentences = tn.normalize_corpus(corpus=sentences_series, text_lower_case=False,
                                     text_lemmatization=False, stopword_removal=False)
print(norm_sentences[:3], '\n')

# starting on page 358
import itertools
stopwords = nltk.corpus.stopwords.words('english')

def get_chunks(sentences, grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',
               stopword_list=stopwords):
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    for sentence in sentences:
        tagged_sents = nltk.pos_tag_sents(
            [nltk.word_tokenize(sentence)])

        chunks = [chunker.parse(tagged_sent)
                  for tagged_sent in tagged_sents]

        wtc_sents = [nltk.chunk.tree2conlltags(chunk)
                     for chunk in chunks]

        flattened_chunks = list(itertools.chain.from_iterable(
            wtc_sent for wtc_sent in wtc_sents)
        )

        valid_chunks_tagged = [(status, [wtc for wtc in chunk]) for status, chunk in
                               itertools.groupby(flattened_chunks, lambda word_pos_chunk:
                                word_pos_chunk[2] != 'O')]

        valid_chunks = [' '.join(word.lower()
                                 for word, tag, chunk in wtc_group
                                    if word.lower() not in stopword_list)
                                        for status, wtc_group in valid_chunks_tagged
                                            if status]

        all_chunks.append(valid_chunks)

    return all_chunks

# page 360
chunks = get_chunks(norm_sentences)
print('Chunks:\n', chunks, '\n')

# page 361
from gensim import corpora, models

def get_tfidf_weighted_keyphrases(sentences, grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',
                                  top_n=10):
    valid_chunks = get_chunks(sentences, grammar=grammar)

    dictionary = corpora.Dictionary(valid_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    weighted_phrases = {dictionary.get(idx): value for doc in corpus_tfidf for idx, value in doc}
    weighted_phrases = sorted(weighted_phrases.items(), key=itemgetter(1), reverse=True)
    weighted_phrases = [(term, round(wt, 3)) for term, wt in weighted_phrases]

    return weighted_phrases[:top_n]

# top 30 tf-idf weighted keyphrases
print('Top 30 TF-IDF keyphrases:]n', get_tfidf_weighted_keyphrases(sentences=norm_sentences, top_n=30), '\n')

# page 362
from gensim.summarization import keywords

# NOTE this code doesn't run in Python 3.7 - one of the sub-packages needs to be updated
# by the maintainers. You can switch to 3.6 or ship these 2 lines.
key_words = keywords(sentences, ratio=1.0, scores=True, lemmatize=False)
print('Gensim\'s summarization model results:\n', [(item, round(score, 3)) for item, score in key_words][:25])