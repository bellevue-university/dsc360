# corrected spellings, starting on page 143
import re, collections

def tokens(text): 
    """
    Get all words from the corpus
    """
    return re.findall('[a-z]+', text.lower()) 

WORDS = tokens(open('big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)
# top 10 words in corpus
print('Top 10 words in corpus:')
print(WORD_COUNTS.most_common(10), '\n')

def edits0(word):
    """
    Return all strings that are zero edits away 
    from the input word (i.e., the word itself).
    """
    return {word}

def edits1(word):
    """
    Return all strings that are one edit away 
    from the input word.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        """
        Return a list of all possible (first, rest) pairs 
        that the input word is made of.
        """
        return [(word[:i], word[i:]) 
                for i in range(len(word)+1)]
                
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    """Return all strings that are two edits away 
    from the input word.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    """
    Return the subset of words that are actually
    in our WORD_COUNTS dictionary.
    """
    return {w for w in words if w in WORD_COUNTS}

print('Input words:')
# input word
word = 'fianlly'

# zero edit distance from input word
print(edits0(word))
# returns null set since it is not a valid word
print(known(edits0(word)))
# one edit distance from input word
print(edits1(word))
# get correct words from above set
print(known(edits1(word)))
# two edit distances from input word
print(edits2(word))
# get correc twords from above set
print(known(edits2(word)))
candidates = (known(edits0(word)) or known(edits1(word)) or known(edits2(word)) or [word])
print(candidates, '\n')

def correct(word):
    '''
    Get the best correct spelling for the input word
    :param word: the input word
    :return: best correct spelling
    '''
    # priority is for edit distance 0, then 1, then 2
    # else defaults to the input word iteself.
    candidates = (known(edits0(word)) or known(edits1(word)) or known(edits2(word)) or [word])
    return max(candidates, key=WORD_COUNTS.get)

print(correct('fianlly'))
print(correct('FIANLLY'), '\n')

def correct_match(match):
    '''
    Spell-correct word in match, and preserve proper upper/lower/title case.
    :param match: word to be corrected
    :return: corrected word
    '''
    word = match.group()
    def case_of(text):
        '''
        Return the case-function appropriate for text: upper/lower/title/as-is
        :param text: The text to be acted on
        :return: Correct text
        '''
        return (str.upper if text.isupper() else
                str.lower if text.islower() else
                str.title if text.istitle() else
                str)
    return case_of(word)(correct(word.lower()))

def correct_text_generic(text):
    '''
    Correct all the words within a text, returning the corrected text
    :param text: Text to be corrected
    :return: Corrected text
    '''
    return re.sub('[a-zA-Z]+', correct_match, text)

print(correct_text_generic('fianlly'))
print(correct_text_generic('FIANLLY'), '\n')

print('TextBlob way (you may need to use pip to install textblob):')
from textblob import Word
w = Word('fianlly')
print(w.correct())
# check suggestions
print(w.spellcheck())
# another example
w = Word('flaot')
print(w.spellcheck())
