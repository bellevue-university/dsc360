from PIL import Image
import pytesseract
import spacy
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
bowers_text = pytesseract.image_to_string(Image.open('C:\\Users\\neugg\\OneDrive\\Documents\\GitHub\\text-analytics-python-neuggs\\week_3\\data\\bowers.jpg'))

bowers_clean = ' '.join(bowers_text.split())
print(bowers_clean)

nlp = spacy.load('en_core_web_sm')
doc = nlp(bowers_clean)

for token in doc:
    print(token.text, token.pos_, token.dep_)