# porter stemmer
from nltk.stem import PorterStemmer
ps = PorterStemmer()

print('Porter stemmer:')
print(ps.stem('jumping'), ps.stem('jumps'), ps.stem('jumped'))
print(ps.stem('lying'))
print(ps.stem('strange'), '\n')

# lancaster stemmer
print('Lancaster stemmer:')
from nltk.stem import LancasterStemmer
ls = LancasterStemmer()
print(ls.stem('jumping'), ls.stem('jumps'), ls.stem('jumped'))
print(ls.stem('lying'))
print(ls.stem('strange'), '\n')

# regex stemmer
print('Regex stemmer:')
from nltk.stem import RegexpStemmer
rs = RegexpStemmer('ing$|s$|ed$', min=4)
print(rs.stem('jumping'), rs.stem('jumps'), rs.stem('jumped'))
print(rs.stem('lying'))
print(rs.stem('strange'), '\n')

# snowball stemmer
print('Snowball stemmer:')
from nltk.stem import SnowballStemmer
ss = SnowballStemmer("german")
print('Supported Languages:', SnowballStemmer.languages)
# autobahnen -> cars
# autobahn -> car
print(ss.stem('autobahnen'))
# springen -> jumping
# spring -> jump
print(ss.stem('springen'), '\n')

# lemmatization
print('WordNet lemmatization:')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
# lemmatize nouns
print(wnl.lemmatize('cars', 'n'))
print(wnl.lemmatize('men', 'n'))
# lemmatize verbs
print(wnl.lemmatize('running', 'v'))
print(wnl.lemmatize('ate', 'v'))
# lemmatize adjectives
print(wnl.lemmatize('saddest', 'a'))
print(wnl.lemmatize('fancier', 'a'))
# ineffective lemmatization
print(wnl.lemmatize('ate', 'n'))
print(wnl.lemmatize('fancier', 'v'), '\n')

print('spaCy:')
import spacy
nlp = spacy.load('en_core_web_sm')
text = 'My system keeps crashing! his crashed yesterday, ours crashes daily'

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-'
                     else word.text for word in text])
    return text

print(lemmatize_text(text))
