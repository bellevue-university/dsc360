# This is the author's normalization file - best to use for this one script
from week_7.normalization import normalize_corpus
import pickle
import pandas as pd

# Load from earlier work
with open("data/bigram_model.txt", "rb") as fp:
    bigram_model = pickle.load(fp)
with open("data/dictionary.txt", "rb") as fp:
    dictionary = pickle.load(fp)
with open("data/best_lda_model.txt", "rb") as fp:
    best_lda_model = pickle.load(fp)
with open("data/topics_df.txt", "rb") as fp:
    topics_df = pickle.load(fp)

# Predicting Topics for New Research Papers - starting on page 415
import glob
new_paper_files = glob.glob('nipstxt/nips16/nips16*.txt')
new_papers = []

for fn in new_paper_files:
    with open(fn, encoding='utf-8', errors='ignore', mode='r+') as f:
        data = f.read()
        new_papers.append(data)

print('Total New Papers:', len(new_papers), '\n')

def text_preprocessing_pipeline(documents, normalizer_fn, bigram_model):
    norm_docs = normalizer_fn(documents)
    norm_docs_bigrams = bigram_model[norm_docs]
    return norm_docs_bigrams

def bow_features_pipeline(tokenized_docs, dictionary):
    paper_bow_features = [dictionary.doc2bow(text) for text in
                          tokenized_docs]
    return paper_bow_features

norm_new_papers = text_preprocessing_pipeline(documents=new_papers, normalizer_fn=normalize_corpus,
                                              bigram_model=bigram_model)
norm_bow_features = bow_features_pipeline(tokenized_docs=norm_new_papers, dictionary=dictionary)

print(norm_new_papers[0][:30], '\n')
print(norm_bow_features[0][:30], '\n')

# page 416
def get_topic_predictions(topic_model, corpus, topn=3):
    topic_predictions = topic_model[corpus]
    best_topics = [[(topic, round(wt, 3))
                    for topic, wt in sorted(topic_predictions[i],
                                            key=lambda row: -row[i])[:topn]]
                    for i in range(len(topic_predictions))]
    return best_topics

# putting the function in action
topic_preds = get_topic_predictions(topic_model=best_lda_model,
                                    corpus=norm_bow_features, topn=2)
print('Topic Predictions\n', topic_preds)

# page 417
results_df = pd.DataFrame()
results_df['Papers'] = range(1, len(new_papers)+1)
results_df['Dominant Topics'] = [[top_num+1 for top_num, wt in item]
                                 for item in topic_preds]
res = results_df.set_index(['Papers'])['Dominant Topics']\
    .apply(pd.Series).stack().reset_index(level=1, drop=True)
results_df = pd.DataFrame({'Dominant Topics': res.values}, index=res.index)
results_df['Contribution %'] = [topic_wt for topic_list in
                                [[round(wt*100, 2) for topic_num, wt in item]
                                 for item in topic_preds]
                                for topic_wt in topic_list]
results_df['Topic Desc'] = [topics_df.iloc[t-1]['Terms per Topic']
                            for t in results_df['Dominant Topics'].values]
results_df['Paper Desc'] = [new_papers[i-1][:200]
                            for i in results_df.index_values]
pd.set_option('display.max_colwidth', 300)
print('Results for each paper\n', results_df, '\n')

