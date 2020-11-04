# named entity recognition starting on page 537
import spacy
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

text = re.sub(r'\n', '', text) # remove extra newlines
nlp = spacy.load('en_core_web_sm')
text_nlp = nlp(text)

# print named entities in article
print('Named entities (spaCy):')
ner_tagged = [(word.text, word.ent_type_) for word in text_nlp]
print(ner_tagged, '\n')

# visualize spacy page 539
from spacy import displacy

# visualize named entities
html = displacy.render(text_nlp, style='ent',)
html_file= open('./chapter_8/page_539_displacy.html','w')
html_file.write(html)
html_file.close()

# Programmatic extraction staring on page 539
named_entities = []
temp_entity_name = ''
temp_named_entity = None
for term, tag in ner_tagged:
    if tag:
        temp_entity_name = ' '.join([temp_entity_name, term]).strip()
        temp_named_entity = (temp_entity_name, tag)
    else:
        if temp_named_entity:
            named_entities.append(temp_named_entity)
            temp_entity_name = ''
            temp_named_entity = None
print('Named entities:\n', named_entities, '\n')

# viewing the top entity types
from collections import Counter
c = Counter([item[1] for item in named_entities])
print('Most common entities:\n', c.most_common(), '\n')