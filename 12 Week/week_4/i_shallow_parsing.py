# starting on page 173
print('treebank:')
from nltk.corpus import treebank_chunk
data = treebank_chunk.chunked_sents()
train_data = data[:3500]
test_data = data[3500:]
print(train_data[7], '\n')

print('regext parser:')
simple_sentence = 'US unveils world\'s most powerful supercomputer, beats China.'
from nltk.chunk import RegexpParser
import nltk
from pattern.en import tag
# get POS tagged sentence
tagged_simple_sent = nltk.pos_tag(nltk.word_tokenize(simple_sentence))
print('POS Tags:', tagged_simple_sent)

chunk_grammar = """
NP: {<DT>?<JJ>*<NN.*>}
"""
rc = RegexpParser(chunk_grammar)
c = rc.parse(tagged_simple_sent)
print(c, '\n')

print('chinking:')
chink_grammar = """
NP: {<.*>+} # chunk everything as NP
}<VBD|IN>+{
"""
rc = RegexpParser(chink_grammar)
c = rc.parse(tagged_simple_sent)
# print and view chunked sentence using chinking
print(c, '\n')

# create a more generic shallow parser
print('more generic shallow parser:')
grammar = """
NP: {<DT>?<JJ>?<NN.*>}  
ADJP: {<JJ>}
ADVP: {<RB.*>}
PP: {<IN>}      
VP: {<MD>?<VB.*>+}
"""
rc = RegexpParser(grammar)
c = rc.parse(tagged_simple_sent)
# print and view shallow parsed simple sentence
print(c)
# Evaluate parser performance on test data
print(rc.evaluate(test_data), '\n')

print('chunked and treebank:')
from nltk.chunk.util import tree2conlltags, conlltags2tree
train_sent = train_data[7]
print(train_sent)
# get the (word, POS tag, Chung tag) triples for each token
wtc = tree2conlltags(train_sent)
print(wtc)
# get shallow parsed tree back from the WTC trples
tree = conlltags2tree(wtc)
print(tree, '\n')

print('NGramTagChunker:')
def conll_tag_chunks(chunk_sents):
  tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
  return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]
  
def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff
  
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI
class NGramTagChunker(ChunkParserI):
  def __init__(self, train_sentences, 
               tagger_classes=[UnigramTagger, BigramTagger]):
    train_sent_tags = conll_tag_chunks(train_sentences)
    self.chunk_tagger = combined_tagger(train_sent_tags, tagger_classes)

  def parse(self, tagged_sentence):
    if not tagged_sentence: 
        return None
    pos_tags = [tag for word, tag in tagged_sentence]
    chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
    chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tags]
    wpc_tags = [(word, pos_tag, chunk_tag) for ((word, pos_tag), chunk_tag)
                     in zip(tagged_sentence, chunk_tags)]
    return conlltags2tree(wpc_tags)

# train the shallow parser
ntc = NGramTagChunker(train_data)
# test parser performance on test data
print(ntc.evaluate(test_data))

# the next 2 lines don't belong and have been commented out
# sentence_nlp = nlp(sentence)
# tagged_sentence = [(word.text, word.tag_) for word in sentence_nlp]

# parse our sample sentence
tree = ntc.parse(tagged_simple_sent)
print(tree)
tree.draw()

print('wall street journal:')
# only need the next line once
#nltk.download('conll2000')
from nltk.corpus import conll2000
wsj_data = conll2000.chunked_sents()
train_wsj_data = wsj_data[:10000]
test_wsj_data = wsj_data[10000:]
print(train_wsj_data[10])

# tran the shallow parser
tc = NGramTagChunker(train_wsj_data)
# test performance on test data
print(tc.evaluate(test_wsj_data))

# there's code on the start of page 183 that's a repeat of the code on 181
# I didn't even write it - no need