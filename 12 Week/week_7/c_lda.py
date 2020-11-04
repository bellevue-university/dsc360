# First, load data from the last section
# NOT in the book
import numpy as np
import pickle

# You need this to avoid a python restart while running the CoherenceModel
def main():
    with open("data/bow_corpus.txt", "rb") as fp:
        bow_corpus = pickle.load(fp)
    with open("data/dictionary.txt", "rb") as fp:
        dictionary = pickle.load(fp)
    with open("data/norm_corpus_bigrams.txt", "rb") as fp:
        norm_corpus_bigrams = pickle.load(fp)

    TOTAL_TOPICS = 10

    import gensim
    # page 391
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary,
                                       chunksize=1740, alpha='auto', eta='auto',
                                       random_state=42, iterations=500,
                                       num_topics=TOTAL_TOPICS, passes=20,
                                       eval_every=None)
    print('LDA Topics with Weights:')
    for topic_id, topic in lda_model.show_topics(num_topics=TOTAL_TOPICS, num_words=20):
        print('Topic #' + str(topic_id+1) + ':')
        print(topic, '\n')

    # page 393
    topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
    avg_coherence_score = np.mean([item[1] for item in topics_coherences])
    print('Avg. Coherence Score:', avg_coherence_score, '\n')

    # page 396
    topics_with_wts = [item[0] for item in topics_coherences]
    print('LDA Topics with Weights')
    print('=' * 50)
    for idx, topic in enumerate(topics_with_wts):
        print('Topic #' + str(idx+1) + ':')
        print([(term, round(wt, 3)) for wt, term in topic], '\n')

    # page 397
    print('LDA Topics without Weights')
    print('=' * 50)
    for idx, topic in enumerate(topics_with_wts):
        print('Topic #' + str(idx+1) + ':')
        print([term for wt, term in topic], '\n')

    # page 399
    cv_coherence_model_lda = \
        gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus,
                                      texts=norm_corpus_bigrams, dictionary=dictionary,
                                      coherence='c_v')
    avg_coherence_cv = cv_coherence_model_lda.get_coherence()

    umass_coherence_model_lda = \
        gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus,
                                     texts=norm_corpus_bigrams, dictionary=dictionary,
                                     coherence='u_mass')
    avg_coherence_umass = umass_coherence_model_lda.get_coherence()

    perplexity = lda_model.log_perplexity(bow_corpus)

    print('Avg. Coherence Score (cv):', avg_coherence_cv)
    print('Avg. Coherence Score (umass):', avg_coherence_umass)
    print('Model Perpelxity:', perplexity, '\n')

# You need this to avoid a python restart while running the CoherenceModel
if __name__ == '__main__':
    main()