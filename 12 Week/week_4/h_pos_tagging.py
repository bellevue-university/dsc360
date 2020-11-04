# starting on page 166
sentence = 'US unveils world\'s most powerful supercomputer, beats China.'
import pandas as pd
import spacy
from pprint import pprint

print('spaCy:')
nlp = spacy.load('en_core_web_sm')
sentence_nlp = nlp(sentence)
# POS tagging with spaCy
spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in sentence_nlp]
# the .T in the book transposes rows and columsn, but it's harder to read
pprint(pd.DataFrame(spacy_pos_tagged, columns=['Word', 'POS tag', 'Tag type']))

# POS tagging with nltk
print('\n', 'nltk')
import nltk
# only need the following two lines one time
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
nltk_pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence), tagset='universal')
pprint(pd.DataFrame(nltk_pos_tagged, columns=['Word', 'POS tag']))

print('\n', 'treebank:')
# you only need the next line once
# nltk.download('treebank')
from nltk.corpus import treebank
data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]
print(train_data[0])

print('\n', 'default tagger:')
# default tagger
from nltk.tag import DefaultTagger
dt = DefaultTagger('NN')
# accuracy on test data
print(dt.evaluate(test_data))
# tagging our sample headline
print(dt.tag(nltk.word_tokenize(sentence)))

print('\n', 'regex tagger')
# regex tagger
from nltk.tag import RegexpTagger
# define regex tag patterns
patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')                     # nouns (default) ... 
]
rt = RegexpTagger(patterns)
# accuracy on test data
print(rt.evaluate(test_data))
# tagging our sample headline
print(rt.tag(nltk.word_tokenize(sentence)))

print('\n', 'n gram taggers')
## N gram taggers
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

ut = UnigramTagger(train_data)
bt = BigramTagger(train_data)
tt = TrigramTagger(train_data)

# testing performance on unigram tagger
print(ut.evaluate(test_data))
print(ut.tag(nltk.word_tokenize(sentence)))

# testing performance of bigram tagger
print(bt.evaluate(test_data))
print(bt.tag(nltk.word_tokenize(sentence)))

# testing performance of trigram tagger
print(tt.evaluate(test_data))
print(tt.tag(nltk.word_tokenize(sentence)))

def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff

ct = combined_tagger(train_data=train_data,
                     taggers=[UnigramTagger, BigramTagger, TrigramTagger],
                     backoff=rt)
print(ct.evaluate(test_data))
print(ct.tag(nltk.word_tokenize(sentence)))

print('\n', 'naive bayes and maxent')
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger
nbt = ClassifierBasedPOSTagger(train=train_data,
                               classifier_builder=NaiveBayesClassifier.train)
print(nbt.evaluate(test_data))
print(nbt.tag(nltk.word_tokenize(sentence)))

# the following takes a LONG time to run - run if you have time
'''
met = ClassifierBasedPOSTagger(train=train_data,
                               classifier_builder=MaxentClassifier.train)
print(met.evaluate(test_data))
print(met.tag(nltk.word_tokenize(sentence)))
'''