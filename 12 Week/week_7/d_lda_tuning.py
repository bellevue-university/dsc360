# First, load data from the last section
# NOT in the book
import numpy as np
import pickle
import gensim
import pandas as pd

# WARNING: this file takes a very long time to run.
# 100%|██████████| 29/29 [1:40:11<00:00, 197.19s/it] - yeah, that's an hour and 40 minutes.

# You need the main() function to avoid a python restart while running the CoherenceModel
def main():
    TOTAL_TOPICS = 10
    with open("data/papers.txt", "rb") as fp:
        papers = pickle.load(fp)
    with open("data/bow_corpus.txt", "rb") as fp:
        bow_corpus = pickle.load(fp)
    with open("data/dictionary.txt", "rb") as fp:
        dictionary = pickle.load(fp)
    with open("data/norm_corpus_bigrams.txt", "rb") as fp:
        norm_corpus_bigrams = pickle.load(fp)

    # LDA Tuning: Finding the Optimal Number of Topics - starting on page 402
    # NOTE the use of multicore to speed things up.
    from tqdm import tqdm
    def topic_model_coherence_generator(corpus, texts, dictionary, start_topic_count=2,
                                        end_topic_count=10, step=1, workers=2):
        models = []
        coherence_scores = []
        for topic_nums in tqdm(range(start_topic_count, end_topic_count + 1, step)):
            lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary,
                                               chunksize=1740,
                                               random_state=42, iterations=500,
                                               num_topics=topic_nums, passes=20,
                                               eval_every=None, workers=workers)
            cv_coherence_model_lda = \
                gensim.models.CoherenceModel(model=lda_model, corpus=corpus,
                                             texts=texts, dictionary=dictionary,
                                             coherence='c_v')
            coherence_score = cv_coherence_model_lda.get_coherence()
            coherence_scores.append(coherence_score)
            models.append(lda_model)
        return models, coherence_scores

    # changed end_topic_count to 20 from 30 to speed things up
    lda_models, coherence_scores = topic_model_coherence_generator(bow_corpus, norm_corpus_bigrams,
                                                                   dictionary, start_topic_count=2,
                                                                   end_topic_count=20, step=1)
    coherence_df = pd.DataFrame({'Number of Topics': range(2, 31, 1),
                                 'Coherence Score': np.round(coherence_scores, 4)})
    print(coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10), '\n')

    # Plot - page 404
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')

    x_ax = range(2, 31, 1)
    y_ax = coherence_scores
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_ax, c='r')
    plt.axhline(y=0.535, c='k', linestyle='--', linewidth=2)
    plt.rcParams['figure.facecolor'] = 'white'
    xl = plt.xlabel('Number of Topics')
    y1 = plt.ylabel('Coherence Score')
    plt.show()

    # page 405
    best_model_idx = coherence_df[coherence_df['Number of Topics'] == 20].index[0]
    best_lda_model = lda_models[best_model_idx]
    print('Number of topics for best LDA model:', best_lda_model.num_topics, '\n')

    topics = [[(term, round(wt, 3)) for term, wt in best_lda_model.show_topic(n, topn=20)]
              for n in range(0, best_lda_model.num_topics)]

    for idx, topic in enumerate(topics):
        print('Topic #' + str(idx+1) + ':')
        print([term for term, wt in topic])
    print('\n')

    # page 407
    topics_df = pd.DataFrame([[term for term in topic] for topic in topics],
                             columns = ['Term' + str(i) for i in range(1, 21)],
                             index=['Topic ' + str(t) for t in range(1, best_lda_model.num_topics+1)]).T
    print(topics_df, '\n')

    # page 408
    pd.set_option('display.max_colwidth', -1)
    topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                              for topic in topics], columns = ['Terms per Topic'],
                             index=['Topic' + str(t)
                                    for t in range(1, best_lda_model.num_topics+1)])
    print(topics_df, '\n')

    # Interpreting model results - starting on page 409
    tm_results = best_lda_model[bow_corpus]
    corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                     for topics in tm_results]
    print('First five topics:', corpus_topics[:5],'\n')

    corpus_topic_df = pd.DataFrame()
    corpus_topic_df['Document'] = range(0, len(papers))
    corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
    corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
    corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic']
                                     for t in corpus_topics]
    corpus_topic_df['Paper'] = papers

    # Dominant Topics Distribution Across Corpus - starting on page 410
    pd.set_option('display.max_colwidth', 200)
    topics_stats_df = corpus_topic_df.groupby('Dominant Topic').agg({'Dominant Topic': {
        'Doc Count': np.size, '% Total Docs': np.size}})
    topics_stats_df = topics_stats_df['Dominant Topic'].reset_index()
    topics_stats_df['% Total Docs'] = topics_stats_df['% Total Docs'].apply(
        lambda row: round((row*100) / len(papers), 2))
    topics_stats_df['Topic Desc'] = [topics_df.iloc[t]['Terms per Topic']
                                     for t in range(len(topics_stats_df))]
    print('Topic Status DF:\n', topics_stats_df)

    # Dominant Topics in Specific Research Papers - page 412
    pd.set_option('display.max_colwidth', 200)
    print(corpus_topic_df[corpus_topic_df['Document'].isin([681, 9, 392, 1622, 17, 906,
                                                       996, 503, 13, 733])], '\n')

    # Relevant Research Papers per Topic Based on Dominance - page 413
    print(corpus_topic_df.groupby('Dominant Topic').apply(
        lambda topic_set: (topic_set.sort_values(by=['Contribution %'],
                                                 ascending=False).iloc[0])))

    # save the best lda and topics df models - you'll need them in the next section
    with open('data/best_lda_model.txt', 'wb') as fp:
        pickle.dump(best_lda_model, fp)
    with open('data/topics_df.txt', 'wb') as fp:
        pickle.dump(topics_df, fp)

# You need this to avoid a python restart while running the CoherenceModel
if __name__ == '__main__':
    main()