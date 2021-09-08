"""
Florian Guillot & Julien Donche: Project 7
 Topic modeling and keywords extractions for Holi.io
 Jedha Full Stack, dsmf-paris-13
 08-2021

FILE NAME : 
topic_text.py

OBJECTIVE : 
This function use a pretrained topic_modeling model to associate topics to a text

INPUTS:
 - 'model'    : a topic modeling model that will associate your text to topics. On our side we have chosen LDA
 - 'article'  : string, text to colorize
 - 'n_topics' : integer, number of topics you want to associate to the text
 
OUTPUTS:
list of topics as string
"""

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

#The topics (bags of words from LDA) are analized by the business and associated and named.
list_topics = ['garbage', 'fooding', 'competition', 'people', 'weather', 'entertainment', 'finance', 'household', 'tourism', 'football', 'law', 'environment', 'university sport', 'cooking', 'urbanism', 'police', 'international', 'education', 'consumption', 'nfl', 'motors', 'basketball', 'baseball', 'health', 'US elections']

# Total number of topics in our model 
topics_number =len(list_topics) 

# We associate each topic_id to a name
dict_topics = {i: list_topics[i] for i in range(topics_number)} 

# -----------------------------------------------------------------------------
# The model
# -----------------------------------------------------------------------------

def topic_text(model, article, n_topics):
    topics = []
    doc = model.id2word.doc2bow(article.split()) # Vectorization with Doc2Bow
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True) # Be careful to have a model with per_word_topics = True 
    for idd, prop in sorted(doc_topics,key=lambda x: x[1], reverse=True)[:n_topics+1]: # We sort the list to have the best topics  at the beginning, and add 1 because we handle the topic "11" which is the trash topic
        if idd != 0: # The topic '0' is a trash topic, as the model has put everything it does not understand in it. We won't use it
            topics.append(dict_topics[idd])
    if len(topics)>n_topics: topics = topics[:n_topics]  
    return topics