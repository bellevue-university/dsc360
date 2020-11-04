sentence = 'US unveils world\'s most powerful supercomputer, beats China.'
import spacy
nlp = spacy.load('en_core_web_sm')
sentence_nlp = nlp(sentence)
dependency_pattern = '{left}<---{word}[{w_type}]--->{right}\n--------'
for token in sentence_nlp:
    print(dependency_pattern.format(word=token.orth_,
                                  w_type=token.dep_,
                                  left=[t.orth_ 
                                            for t 
                                            in token.lefts],
                                  right=[t.orth_ 
                                             for t 
                                             in token.rights]))
                                             
# if you're using Jupyter notebook, this is much easier, just
# follow the code in teh book on page 187.
from spacy import displacy
html = displacy.render(sentence_nlp,
                       style='dep',
                       options={'distance': 100,
                                'arrow_stroke': 1.5,
                                'arrow_width': 8})
html_file= open('page_187_displacy.html','w')
html_file.write(html)
html_file.close()

