# starting on page 106
import nltk
import matplotlib.pyplot as plt
import re

bible = nltk.corpus.gutenberg.sents('bible-kjv.txt')
print('First 5 sentences, tokenized:\n', bible[:5], '\n')

print('Length:\n', len(bible), '\n')
# no need to strip new lines, this is already done!

# the histogram is different because of how the corpus is structured now
line_lengths = [len(sent) for sent in bible]
h = plt.hist(line_lengths)
plt.show()

# no need to tokenize, this is also already done!

# page 108, bottom same plot as before
total_tokens_per_line = [len(sent) for sent in bible]
h = plt.hist(total_tokens_per_line, color='orange')
plt.show()

# page 109
words = nltk.corpus.gutenberg.words('bible-kjv.txt')
print('First 20 words\n', words[:20])
words = list(filter(None, [re.sub(r'[^A-Za-z]', '', word) for word in words]))
print('Cleaned words\n', words[:20], '\n')

# page 110
from collections import Counter
words = [word.lower() for word in words]
c = Counter(words)
print('Most common words:\n', c.most_common(10), '\n')

stopwords = nltk.corpus.stopwords.words('english')
words = [word.lower() for word in words if word.lower() not in stopwords]
c = Counter(words)
print('Most common words, no stop words\n', c.most_common(10))