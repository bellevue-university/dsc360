import nltk

# Expanding Contractions - starting on page 136
from chapter_3.c_contractions import CONTRACTION_MAP
import re

def expand_contractions(sentence, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expaned_text = contractions_pattern.sub(expand_match, sentence)
    expanded_text = re.sub("'", "", expaned_text)
    return expanded_text
print('Exanding contractions:')
print(expand_contractions("Y'all can't expand contractions I'd think"), '\n')

# Removing special characters, page 138
def remove_special_characters(text, remove_digits =False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

print('Remove special characters:')
print(remove_special_characters('Well this was fun! What do you thin? 123#@!', remove_digits=True), '\n')

# case conversions
print('Case conversions:')
# lowercase
text = 'The quick brown fox jumped over The Big Dog'
print(text.lower())
# uppercase
print(text.upper())
# title case
print(text.title(), '\n')

# text correction
# correcting repeating characters - pages 139-140
old_word = 'finalllyyy'
repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
match_substitution = r'\1\2\3'
step = 1
while True:
    # remove on repeated character
    new_word = repeat_pattern.sub(match_substitution, old_word)
    if new_word != old_word:
        print('Step: {} Word: {}'.format(step, new_word))
        step += 1 #update step
        # update old word to last substituted state
        old_word = new_word
        continue
    else:
        print('Final word: ', new_word, '\n')
        break

# pages 140-141
print('Wordnet:')
from nltk.corpus import wordnet
old_word = 'finalllyyy'
repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
match_substitution = r'\1\2\3'
step = 1
while True:
    # check for semantically correct words
    if wordnet.synsets(old_word):
        print('Final correct word: ', old_word, '\n')
        break
    # remove on repeated characters
    new_word = repeat_pattern.sub(match_substitution, old_word)
    if new_word != old_word:
        print('Step: {} Word: {}'.format(step, new_word))
        step += 1  # update step
        # update old word to last substituted state
        old_word = new_word
        continue
    else:
        print('Final word: ', new_word, '\n')
        break

# pages 141-142
def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

sample_sentence = 'My schooool is realllllyyy amaaazingggg'
correct_tokens = remove_repeated_characters(nltk.word_tokenize(sample_sentence))
print(' '.join(correct_tokens), '\n')