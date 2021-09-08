"""
Florian Guillot & Julien Donche: Project 7
 Topic modeling and keywords extractions for Holi.io
 Jedha Full Stack, dsmf-paris-13
 08-2021

This project is the final project as Jedha Students. 
Idea has been submitted by Holi.io Founder : Clément Sirvente
The specifications from Holi.io can be found [here](https://github.com/FlorianG-dev/Jedha_certification/blob/master/7_Holi/Project_initialization.pdf). 
It is the projet number 1 : Topic modeling

FILE NAME : app.py
This script is used to lauch our Application Flask
"""

# -----------------------------------------------------------------------------
#  Initialization
# -----------------------------------------------------------------------------

# We import libraries
from flask import Flask, request, render_template, Markup, jsonify
import joblib
# We import the functions we have built, they have as parameters
# - The text
# - Model
# - Application parameters (number & lenght of keywords, number of topics)
from functions.find_keywords import find_keywords
from functions.preprocessing_text import preprocess_text
from functions.topic_text import topic_text
from functions.color_text import color_text
from functions.color_text import mycolors


#  We call & name the app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1 # infinite

#  We load the model
model = joblib.load("models/lda_model.joblib")


# -----------------------------------------------------------------------------
#  We build the app page after page
# -----------------------------------------------------------------------------


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-HOME PAGE
@app.route("/")
def index():
     return render_template('index.html')

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-MODEL PAGE
# Our model plot with the help of the pyLDAvis library, see the notebook 'Step2' at the root of the project
@app.route("/model")
def goto_model():
     return render_template('lda.html')

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-API DOCUMENTATION
@app.route("/api_doc")
def goto_api():
     return render_template('api_doc.html')

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-LIVE APPLICATION
@app.route("/live_app")
def goto_live_app():
     return render_template('live_app.html')
 
 
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-LIVE APPLICATION RESULTS PAGE
@app.route("/result_live", methods=['POST'])
def result_live():
     text = request.form['input_text'] # From the form front-end
     l_keywords = int(request.form['l_keywords'][0]) 
     n_keywords = int(request.form['n_keywords'][0])
     n_topics = int(request.form['n_topics'][0])
     
     # We preprocess the text given by the user
     processed_text = preprocess_text(text)
     
     # We extract the keywords
     try:
          keywords = find_keywords(processed_text, l_keywords, n_keywords)
     except:
          keywords=["No keyword found"]
          
     # We make the topic modeling
     try:
          topics = topic_text(model,processed_text, n_topics)
          topics_colored = ["<font color=\"" + mycolors[i] + "\">" + topics[i] + ' ' + "</font>" for i in range(len(topics))]
     except:
          topics = ["No topic found"]
          
     # We colorize the text accordingly with the topics found
     try:
          colored_text = color_text(model,processed_text,n_topics)
     except:
          colored_text = ["Impossible to colorize this text"]
     
     # We return the results 
     return render_template('live_app.html', topics=[Markup(x) for x in topics_colored], keywords=keywords, colored_text = Markup(colored_text))


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-LIVE API
@app.route("/api", methods=["POST"])
def api():
     # Get the data
     req = request.get_json()
     # Check mandatory key
     if "text" in req.keys():
          text = req["text"]
     if "l_keywords" in req.keys():
          l_keywords = int(req["l_keywords"])
     if "n_keywords" in req.keys():
          n_keywords = int(req["n_keywords"])
     if "n_topics" in req.keys():
          n_topics = int(req["n_topics"])
          
     # Preprocessing
     processed_text = preprocess_text(text)
     print('Preprocessing done')
     
     # We extract the the keywords
     try:
        keywords = find_keywords(processed_text, l_keywords, n_keywords)
     except:
        keywords=["No keyword found"]
     print('Preprocessing done')
     
     # We make the topic modeling
     try:
          topics = topic_text(model,processed_text, n_topics)
     except:
          topics = ["No topic found"]   
     print('Topics modeling done')

     return jsonify({"keywords": keywords, "topics":topics}), 200


# -----------------------------------------------------------------------------
#  Bottom activator
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)