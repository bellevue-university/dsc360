# page 118
import requests

data = requests.get('http://www.gutenberg.org/cache/epub/8001/pg8001.html')
content = data.content
# the text that prints is a little different because of book version differences
print(content[1163:2200], '\n')

# pages 118-119
import re
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)

    return stripped_text

clean_content = strip_html_tags(content)
print(clean_content[1163:2045], '\n')