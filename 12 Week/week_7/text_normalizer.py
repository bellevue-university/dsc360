import re
from bs4 import BeautifulSoup
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP

class TextNormalizer():

    def __init__(self):
        print('Starting TextNormalizer')

    def normalize_corpus(self, corpus, html_stripping=True, contraction_expansion=True,
                         accented_char_removal=True, text_lower_case=True, text_lemmatization=True,
                         special_char_removal=True, stopword_removal=True, remove_digits=True):
        # remove carriage return line feeds
        return_corpus = corpus

        if html_stripping:
            return_corpus = corpus.apply(lambda x: self.strip_html_tags(x))
            print('Done strip')
        if text_lower_case:
            return_corpus = return_corpus.apply(lambda x: x.lower())
            print('Done lower')
        if stopword_removal:
            return_corpus = return_corpus.apply(lambda x: self.remove_stopwords(x))
            print('Done stopword')
        if accented_char_removal:
            return_corpus = return_corpus.apply(lambda x: self.remove_accented_chars(x))
            print('Done char remove')
        if contraction_expansion:
            return_corpus = return_corpus.apply(lambda x: self.expand_contractions(x))
            print('Done contract exp')
        if text_lemmatization:
            return_corpus = return_corpus.apply(lambda x: self.lemmatize_text(x))
            print('Done text lemm')
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            return_corpus = return_corpus.apply(lambda x: special_char_pattern.sub(" \\ ", x))
            return_corpus = return_corpus.apply(lambda x:
                                                self.remove_special_characters(x, remove_digits=remove_digits))
            print('Done spec char remove')

        # remove extra whitespace
        return_corpus = return_corpus.apply(lambda x: re.sub(' +', ' ', x))

        return return_corpus

    def strip_html_tags(self, text):
        soup = BeautifulSoup(str(text), 'html.parser')
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = stripped_text.replace('\n\n', ' ')
        stripped_text = stripped_text.replace('\n', ' ')
        stripped_text = stripped_text.replace('\t', ' ')
        return stripped_text

    def expand_contractions(self, sentence, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        expaned_text = contractions_pattern.sub(expand_match, sentence)
        expanded_text = re.sub("'", "", expaned_text)
        return expanded_text

    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii',
                                                          'ignore').decode('utf-8', 'ignore')
        return text

    def lemmatize_text(self, text):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemm_text = lemmatizer.lemmatize(text)
        return lemm_text

    def remove_special_characters(self, text, remove_digits=False):
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
        text = re.sub(pattern, '', text)
        return text

    def remove_stopwords(self, text):
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text
