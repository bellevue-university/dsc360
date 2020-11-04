# Understinding sysnets starting on page 522
from nltk.corpus import wordnet as wn
import pandas as pd

term = 'fruit'
synsets = wn.synsets(term)

print('Total Synsets:', len(synsets), '\n')

pd.options.display.max_colwidth = 200
fruit_df = pd.DataFrame([{'Synset': synset,
                          'Part of Speech': synset.lexname(),
                          'Definition': synset.definition(),
                          'Lemmas': synset.lemma_names(),
                          'Examples': synset.examples()} for synset in synsets])

fruit_df = fruit_df[['Synset', 'Part of Speech', 'Definition', 'Lemmas', 'Examples']]
print('Fruit DataFrame:\n', fruit_df, '\n')

# entailments page 524
print('Entailments:')
for action in ['walk', 'eat', 'digest']:
    action_syn = wn.synsets(action, pos='v')[0]
    print(action_syn, '-- entails -->', action_syn.entailments())
print('\n')

# homonyms\homographs page 524
print('Homonyms/homographs:')
for synset in wn.synsets('bank'):
    print(synset.name(),'-',synset.definition())
print('\n')

# synonyms and antonyms page 525
print('Synonyms and antonyms:')
term = 'large'
synsets = wn.synsets(term)
adj_large = synsets[1]
adj_large = adj_large.lemmas()[0]
adj_large_synonym = adj_large.synset()
adj_large_antonym = adj_large.antonyms()[0].synset()

print('Synonym:', adj_large_synonym.name())
print('Definition:', adj_large_synonym.definition())
print('Antonym:', adj_large_antonym.name())
print('Definition:', adj_large_antonym.definition(), '\n')

term = 'rich'
synsets = wn.synsets(term)[:3]
for synset in synsets:
    rich = synset.lemmas()[0]
    rich_synonym = rich.synset()
    rich_antonym = rich.antonyms()[0].synset()
    print('Synonym:', rich_synonym.name())
    print('Definition:', rich_synonym.definition())
    print('Antonym:', rich_antonym.name())
    print('Definition:', rich_antonym.definition())
print('\n')

# hyponyms and hypernyms page 527
print('Hyponyms and hypernyms:')
term = 'tree'
synsets = wn.synsets(term)
tree = synsets[0]
print('Name:', tree.name())
print('Definition:', tree.definition())

hyponyms = tree.hyponyms()
print('Total Hyponyms:', len(hyponyms))
print('Sample Hyponyms')
for hyponym in hyponyms[:10]:
    print(hyponym.name(), '-', hyponym.definition())

print('\n')
    
hypernyms = tree.hypernyms()
print(hypernyms)

hypernym_paths = tree.hypernym_paths()
print('Total Hypernym paths:', len(hypernym_paths))

print('Hypernym Hierarchy')
print(' -> '.join(synset.name() for synset in hypernym_paths[0]))
print('\n')

# holonyms and meronyms page 529
# member holonyms
print('Holonyms and meronyms:')
member_holonyms = tree.member_holonyms()    
print('Total Member Holonyms:', len(member_holonyms))
print('Member Holonyms for [tree]:-')
for holonym in member_holonyms:
    print(holonym.name(), '-', holonym.definition())
print('\n')

# part meronyms page 529
print('Part meronyms:')
part_meronyms = tree.part_meronyms()
print('Total Part Meronyms:', len(part_meronyms))
print('Part Meronyms for [tree]:-')
for meronym in part_meronyms:
    print(meronym.name(), '-', meronym.definition())
print('\n')

# substance meronyms
print('Substance meronyms:')
substance_meronyms = tree.substance_meronyms()    
print('Total Substance Meronyms:', len(substance_meronyms))
print('Substance Meronyms for [tree]:-')
for meronym in substance_meronyms:
    print(meronym.name(), '-', meronym.definition())
print('\n')

# semantic relationships and similarities starting on page 530
print('Semantic relationships:')
tree = wn.synset('tree.n.01')
lion = wn.synset('lion.n.01')
tiger = wn.synset('tiger.n.02')
cat = wn.synset('cat.n.01')
dog = wn.synset('dog.n.01')

entities = [tree, lion, tiger, cat, dog]
entity_names = [entity.name().split('.')[0] for entity in entities]
entity_definitions = [entity.definition() for entity in entities]

for entity, definition in zip(entity_names, entity_definitions):
    print(entity, '-', definition)
print('\n')

print('Common hypernyms:')
common_hypernyms = []
for entity in entities:
    # get pairwise lowest common hypernyms
    common_hypernyms.append([entity.lowest_common_hypernyms(compared_entity)[0]
                            .name().split('.')[0] for compared_entity in entities])
# build pairwise lower common hypernym matrix
common_hypernym_frame = pd.DataFrame(common_hypernyms,
                                     index=entity_names, 
                                     columns=entity_names)
print(common_hypernym_frame, '\n')

# page 533
similarities = []
for entity in entities:
    # get pairwise similarities
    similarities.append([round(entity.path_similarity(compared_entity), 2)
                         for compared_entity in entities])        
# build pairwise similarity matrix                             
similarity_frame = pd.DataFrame(similarities,
                                index=entity_names, 
                                columns=entity_names)
                                     
print('Similarity frame:\n', similarity_frame, '\n')

# word sense disambiguation starting on page 534
from nltk.wsd import lesk
from nltk import word_tokenize
samples = [('The fruits on that plant have ripened', 'n'),
           ('He finally reaped the fruit of his hard work as he won the race', 'n')]

# perform word sense disambiguation
word = 'fruit'
for sentence, pos_tag in samples:
    word_syn = lesk(word_tokenize(sentence.lower()), word, pos_tag)
    print('Sentence:', sentence)
    print('Word synset:', word_syn)
    print('Corresponding defition:', word_syn.definition())
    print('\n')

samples = [('Lead is a very soft, malleable metal', 'n'),
           ('John is the actor who plays the lead in that movie', 'n'),
           ('This road leads to nowhere', 'v')]

# perform word sense disambiguation
word = 'lead'
for sentence, pos_tag in samples:
    word_syn = lesk(word_tokenize(sentence.lower()), word, pos_tag)
    print('Sentence:', sentence)
    print('Word synset:', word_syn)
    print('Corresponding defition:', word_syn.definition())
    print('\n')
