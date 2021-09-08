"""
Florian Guillot & Julien Donche: Project 7
 Topic modeling and keywords extractions for Holi.io
 Jedha Full Stack, dsmf-paris-13
 08-2021

FILE NAME : 
color_text.py

OBJECTIVE : 
This function colors important words in a text accordingly with the topics a topic_modeling model has found

INPUTS:
 - 'model'    : a topic modeling model that will associate your text to topics. On our side we have chosen LDA
 - 'article'  : string, text to colorize
 - 'n_topics' : integer, number of topics you want to associate to the text
 
OUTPUTS:
The text, as string, with the HTML colors directly writen. For instance : 
<font color="#0ff1ce">this text is green</font>
"""

mycolors = ['#FF0000', '#dde031', '#892ed9', '#0ff1ce', '#ff7518', '#ff6699']

def color_text(model, article, n_topics):
    doc = model.id2word.doc2bow(article.split()) # We vectorize the input
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True) # Be careful to have, when you train tour model, the option per_word_topics=True, if not this won't work
    topics_used = sorted(doc_topics,key=lambda x: x[1], reverse=True)[:n_topics] # We get the n_topics associated to the text
    topic_colors = { topics_used[i][0] : mycolors[i] for i in range(n_topics)} # And associate each topic with a color
    text="" # Initiation of our final text
    for word, topics in word_topics:
        try:
            text+="<font color=\"" + topic_colors[topics[0]] + "\">" + model.id2word[word] + ' ' + "</font>" # We write in 'text' the word with the HTML code needed to color it
        except: 
            text+="<font color=\"#ffffff\">" + model.id2word[word] + ' ' + "</font>" # If the word does not have topics affiliated, we print it white  
    return text