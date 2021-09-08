"""
Florian Guillot & Julien Donche: Project 7
 Topic modeling and keywords extractions for Holi.io
 Jedha Full Stack, dsmf-paris-13
 08-2021

FILE NAME : 
preprocessing_text.py

OBJECTIVE : 
This function will have a text as input and will give a preprocessized text as output, for instance without URL, without stop words, etc...
The process is the same as the one you can find in the notebook 'Step1'

INPUTS:
 - 'article' : string, text to analize
 
OUTPUTS:
processed text, as string
"""


# -----------------------------------------------------------------------------
#  Initialization
# -----------------------------------------------------------------------------

import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm



# -----------------------------------------------------------------------------
#  Cleaning function
# -----------------------------------------------------------------------------

def clean (article):
    
  # We force the utf-8 encoding
  article.encode("utf-8").decode("utf-8")
    
  # Deletes urls
  article = re.sub(r"https:[A-Za-z0-9]+", "", article)
  article = re.sub(r"http:[A-Za-z0-9]+", "", article)
  article = re.sub(r"www\.[A-Za-z0-9]+", "", article)
  
  # We delete the \r and the \n
  article = re.sub(r"\\r|\\n", "", article)
  
  # We delete the 's
  article = re.sub(r"'s", "", article)
  
  # We delete everything that is not alphabetic
  pattern = re.compile(r'[^a-zA-Z]+')
  article = pattern.sub(' ', article)
  
  # Transform multiples spaces in one space
  article = re.sub(r"\s{2,}", " ", article)

  # strip 
  article = article.strip()

  return article

# -----------------------------------------------------------------------------
#  Words to replace or delete, based on observations
# -----------------------------------------------------------------------------

to_replace={
    'sen':'senate',
    'senator':'senate',
    'teacher':'teaching'
}
add_stop_words={'tonight',
                'yes',
                'no',
                'hey',
                'okay',
                'etc',
                'mr',
                'mss',
                'ms',
                'er',
                'v',
                'monthly',
                'tb',
                'sec',
                'mind'}
STOP_WORDS |= add_stop_words

# -----------------------------------------------------------------------------
#  Function to replace or delete those words, based on observations, and everything that is not a noun, adjective or verb
# -----------------------------------------------------------------------------
def text_to_nlp_ready (article):
    nlp = en_core_web_sm.load()
    excluded_tags = {"ADV", "ADP", "AUX", "NUM"}
    article_tokenized = [ token.lemma_ for token in nlp(article) if (token.pos_ not in excluded_tags) & (token.lemma_.lower() not in STOP_WORDS) & (len(token.lemma_) >1) ]
    article_nlp_ready = ' '.join(article_tokenized).lower()
    
    return article_nlp_ready
# -----------------------------------------------------------------------------
#  Final function that will concatenate the different functions above & create a text NLP ready
# -----------------------------------------------------------------------------

def preprocess_text(article):
    return text_to_nlp_ready((clean(article)))