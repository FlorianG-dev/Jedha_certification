
'''
# Florian Guillot : Project 6
### Wine o meter
### Jedha Full Stack, dsmf-paris-13
### 08-2021

Predict and deliver the wine'squality (feature = chimical components, target = quality) trough an API and build a web-app

the app was previously at the adresse : 
https://wine-o-meter-vflo.herokuapp.com/

It has been depreciated
'''

# -----------------------------------------------------------------------------
#  We import libraries
# -----------------------------------------------------------------------------
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from Functions import *

# -----------------------------------------------------------------------------
#  We call and name our app
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
#  We load the model
# -----------------------------------------------------------------------------
classifier = joblib.load("quality_classifier.joblib")

# -----------------------------------------------------------------------------
#  We code the calls for each page of our app
# -----------------------------------------------------------------------------

# The documentation
@app.route('/doc')
def doc():
    return render_template ('doc.html')

# Our home page which will predict in live
@app.route("/")
def home():
    return render_template('index.html')


# Our page which will do the prediction live
@app.route('/predict_live',methods=['POST'])
def predict_live():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text='We predict a quality of this wine around :{}/10'.format(output))



# Our page which will predict through an API
@app.route("/predict", methods=["POST"])
def index():
    # Get the data
    req = request.get_json()
    # Check if input has the right shape and type
    if not POST_error(req)[0]: # Function to determine if the input shape is usable for our model, see the .py Function
        print('out of POST_error')
        print('classifier defined')
        # Predict
        prediction = classifier.predict( [req[key] for key in req.keys()] )
        print('prediction done')
        # Return the result as JSON but first we need to transform the
        # result so as to be serializable by jsonify()
        prediction = [ str(round(prediction[i],0)) for i in range(len(prediction)) ] 
        return jsonify({"predict": prediction, "msg": POST_error(req)[1]}), 200
    return jsonify({"msg": POST_error(req)[1]}), 400 #We give an error message and the code 400 if our function POST_error is True

# -----------------------------------------------------------------------------
#  If we call doirectly the script it will launch the app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)