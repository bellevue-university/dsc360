import nltk
import pandas as pd
import re

text = """Three more countries have joined an “international grand committee” of 
parliaments, adding to calls for Facebook’s boss, Mark Zuckerberg, to give evidence 
on misinformation to the coalition. Brazil, Latvia and Singapore bring the total to 
eight different parliaments across the world, with plans to send representatives to 
London on 27 November with the intention of hearing from Zuckerberg. Since the 
Cambridge Analytica scandal broke, the Facebook chief has only appeared in front of 
two legislatures: the American Senate and House of Representatives, and the European 
parliament. Facebook has consistently rebuffed attempts from others, including the 
UK and Canadian parliaments, to hear from Zuckerberg. He added that an article in 
the New York Times on Thursday, in which the paper alleged a pattern of behavior 
from Facebook to “delay, deny and deflect” negative news stories, “raises further 
questions about recent data breaches were allegedly dealt with within Facebook.”
"""

text = re.sub(r'\n', '', text)
tokenized_text = nltk.tokenize.sent_tokenize(text)

for sentence in tokenized_text:
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    named_ent = nltk.ne_chunk(tagged, binary=False)
    print(named_ent)