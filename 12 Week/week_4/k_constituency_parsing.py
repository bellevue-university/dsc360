sentence = 'US unveils world\'s most powerful supercomputer, beats China.'

'''
This Standford parser is depricated and requires a local server (too complicated for this).
Therefore, I commented all the code out - it's just another parser and does the same thing 
as the rest of the code without the hassle.
# set java path
import os
java_path = r'C:\Program Files\Java\jdk1.8.0_102\bin\java.exe'
os.environ['JAVAHOME'] = java_path
from nltk.parse.stanford import StanfordParser

scp = StanfordParser(path_to_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser.jar',
                   path_to_models_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar')
result = list(scp.raw_parse(sentence))
print(result[0])
result[0].draw()
'''
# starting on page 195
import nltk
from nltk.grammar import Nonterminal
from nltk.corpus import treebank
training_set = treebank.parsed_sents()
print(training_set[1], '\n')

# extract the productions for all annotated training sentences
treebank_productions = list(
                        set(production 
                            for sent in training_set  
                            for production in sent.productions()
                        )
                    )
# view some production rules
print(treebank_productions[0:10])
  
# add productions for each word, POS tag
for word, tag in treebank.tagged_words():
    t = nltk.Tree.fromstring( "("+ tag + " " + word  + ")")
    for production in t.productions():
        treebank_productions.append(production)

# build the PCFG based grammar  
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'), 
                                         treebank_productions)

# build the parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)
# get sample sentence tokens
tokens = nltk.word_tokenize(sentence)
# get parse tree for sample sentence
# this next lines throw and error (see the text on page 197)
# result = list(viterbi_parser.parse(tokens))

# get tokens and their POS tags and check it
tagged_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
print(tagged_sent, '\n')

# extend productions for sample sentence tokens
for word, tag in tagged_sent:
    t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
    for production in t.productions():
        treebank_productions.append(production)

# rebuild grammar
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'), treebank_productions)
# rebuild parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)
# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))
#print parse tree
print(result[0])
# visualize parse tree
result[0].draw()
