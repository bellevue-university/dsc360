# Automated Document Summarization - starting on page 436
# I took the description from:
# Wikipedia. (2020). The Elder Scross V: Skyrim. https://en.wikipedia.org/wiki/The_Elder_Scrolls_V:_Skyrim
DOCUMENT = """
The Elder Scrolls V: Skyrim is an action role-playing video game developed by Bethesda Game Studios and published by 
ethesda Softworks. It is the fifth main installment in The Elder Scrolls series, following The Elder Scrolls IV: 
Oblivion, and was released worldwide for Microsoft Windows, PlayStation 3, and Xbox 360 on November 11, 2011.

The game's main story revolves around the player's character, the Dragonborn, on their quest to defeat Alduin the 
World-Eater, a dragon who is prophesied to destroy the world. The game is set 200 years after the events of Oblivion 
and takes place in Skyrim, the northernmost province of Tamriel. Over the course of the game, the player completes 
quests and develops the character by improving skills. The game continues the open-world tradition of its 
predecessors by allowing the player to travel anywhere in the game world at any time, and to ignore or postpone the 
main storyline indefinitely.

Skyrim was developed using the Creation Engine, rebuilt specifically for the game. The team opted for a unique and 
more diverse open world than Oblivion's Imperial Province of Cyrodiil, which game director and executive producer 
Todd Howard considered less interesting by comparison. The game was released to critical acclaim, with reviewers 
particularly mentioning the character advancement and setting, and is considered to be one of the greatest video 
games of all time. Nonetheless it received some criticism, predominantly for its melee combat and numerous 
technical issues present at launch. The game shipped over seven million copies to retailers within the first week 
of its release, and over 30 million copies on all platforms as of November 2016, making it one of the best selling 
video games in history.

Three downloadable content (DLC) add-ons were released—Dawnguard, Hearthfire, and Dragonborn—which were repackaged 
into The Elder Scrolls V: Skyrim – Legendary Edition and released in June 2013. The Elder Scrolls V: Skyrim – 
Special Edition is a remastered version of the game released for Windows, Xbox One, and PlayStation 4 in October 
2016. It includes all three DLC expansions and a graphical upgrade, along with additional features such as modding 
capabilities on consoles. Versions were released in November 2017 for the Nintendo Switch and PlayStation VR, and 
a stand-alone virtual reality (VR) version for Windows was released in April 2018. These versions were based on 
the remastered release, but the Switch version's graphics upgrade was relative to its hardware capabilities, and 
it did not include the modding features.
"""

# page 438
import re
DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
DOCUMENT = DOCUMENT.strip()

from gensim.summarization import summarize
print('Summarized document:\n', summarize(DOCUMENT, ration=0.2, split=False), '\n')
print('Limited document summary\n', summarize(DOCUMENT, word_count=75, split=False), '\n')

# Text Wrangling - page 439
import nltk
import numpy as np

stop_words = nltk.corpus.stopwords.words('english')