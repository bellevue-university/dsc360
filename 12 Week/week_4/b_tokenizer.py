import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
from pprint import pprint
import numpy as np

## SENTENCE TOKENIZATION
# loading text corpora
alice = gutenberg.raw(fileids='carroll-alice.txt')

# Shorter text than the text, but this is the same core code, which
# comes after showing some data about the Alice corpus.
sample_text = 'We will discuss briefly about the basic syntax,\
 structure and design philosophies. \
 There is a defined hierarchical syntax for Python code which you should remember \
 when writing code! Python is a really powerful programming language!'
print('Sample text: ', sample_text, '\n')

# Total characters in Alice in Wonderland
print('Length of alice: ', len(alice))
# First 100 characters in the corpus
print('First 100 chars of alice: ', alice[0:100], '\n')

## default sentence tokenizer
default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)
sample_sentences = default_st(text=sample_text)
print('Default sentence tokenizer')
print('Total sentences in sample_text:', len(sample_sentences))
print('Sample text sentences :-')
pprint(sample_sentences)
print('\nTotal sentences in alice:', len(alice_sentences))
print('First 5 sentences in alice:-')
pprint(alice_sentences[0:5])

## Other languages sentence tokenization
nltk.download('europarl_raw')
from nltk.corpus import europarl_raw
german_text = europarl_raw.german.raw(fileids='ep-00-01-17.de')
print('Other language tokenization')
# Total characters in the corpus
print(len(german_text))
# First 100 characters in the corpus
print(german_text[0:100])

# default sentence tokenizer 
german_sentences_def = default_st(text=german_text, language='german')
# loading german text tokenizer into a PunktSentenceTokenizer instance
german_tokenizer = nltk.data.load(resource_url='tokenizers/punkt/german.pickle')
german_sentences = german_tokenizer.tokenize(german_text)

# verify the type of german_tokenizer
# should be PunktSentenceTokenizer
print('German tokenizer type:', type(german_tokenizer))

# check if results of both tokenizers match
# should be True
print(german_sentences_def == german_sentences)
# print(first 5 sentences of the corpus
for sent in german_sentences[0:5]:
    print(sent)
print('\n')

## using PunktSentenceTokenizer for sentence tokenization
print('Punkt tokenizer')
punkt_st = nltk.tokenize.PunktSentenceTokenizer()
sample_sentences = punkt_st.tokenize(sample_text)
pprint(np.array(sample_sentences))
print('\n')

## using RegexpTokenizer for sentence tokenization
print('Regex tokenizer')
SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'
regex_st = nltk.tokenize.RegexpTokenizer(
            pattern=SENTENCE_TOKENS_PATTERN,
            gaps=True)
sample_sentences = regex_st.tokenize(sample_text)
# again, the output is different because the sample sentence is different
pprint(sample_sentences)
print('\n')
        
## WORD TOKENIZATION
sentence = "The brown fox wasn't that quick and he couldn't win the race"
# default word tokenizer
print('Word tokenizer')
default_wt = nltk.word_tokenize
words = default_wt(sentence)
print(words, '\n')

# treebank word tokenizer
print('Treebank tokenizer')
treebank_wt = nltk.TreebankWordTokenizer()
words = treebank_wt.tokenize(sentence)
print(words, '\n')

# toktok tokenizer
print('TokTok tokenizer')
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
words = tokenizer.tokenize(sample_text)
print(np.array(words), '\n')

# regex word tokenizer
print('RegEx word tokenizer')
TOKEN_PATTERN = r'\w+'        
regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN,
                                gaps=False)
words = regex_wt.tokenize(sentence)
print(words)

GAP_PATTERN = r'\s+'        
regex_wt = nltk.RegexpTokenizer(pattern=GAP_PATTERN,
                                gaps=True)
words = regex_wt.tokenize(sentence)
print(words)

word_indices = list(regex_wt.span_tokenize(sentence))
print(word_indices)
print([sentence[start:end] for start, end in word_indices], '\n')

# derived regex tokenizers
print("Derived RegEx tokenizers")
wordpunkt_wt = nltk.WordPunctTokenizer()
words = wordpunkt_wt.tokenize(sentence)
print(words, '\n')

whitespace_wt = nltk.WhitespaceTokenizer()
words = whitespace_wt.tokenize(sentence)
print(words, '\n')

# pages 132 - 134
print('Robust tokenizer - NLTK')
def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens

sents = tokenize_text(sample_text)
print(np.array(sents),'\n')

words = [word for sentence in sents for word in sentence]
print(np.array(words), '\n')

print('spaCy...')
import spacy
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
text_spacy = nlp(sample_text)
print(np.array(list(text_spacy.sents)), '\n')

sent_words = [[word for word in sent] for sent in sents]
print(np.array(sent_words), '\n')

# in spacy documentation, this is usually written as [token for token in doc]
words = [word for word in text_spacy]
print(np.array(words), '\n')

# page 135
import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii',
                                                      'ignore').decode('utf-8', 'ignore')
    return text

print(remove_accented_chars('Sòme Åccentềd cliché façades'))

