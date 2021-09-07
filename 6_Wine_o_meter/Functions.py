#list of the different features we seek to predict
features_list = ['fixed acidity', 'volatile acidity', 'citric acid','residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# -----------------------------------------------------------------------------
#  Function to test what we get as a POST request in the API
# -----------------------------------------------------------------------------

import numbers
def POST_error(req): #req is a list of lists
    for index,input in req.items():
        if len(input) !=11 :
            return (True,f"Apparently the prediction with id {index} has not 11 features, please provide 11 features")
        if not all(isinstance(x, numbers.Number) for x in input):
            return (True,f"Apparently in the prediction with id {index} there is a not digit feature, please provide only digit")
    return (False, 'After checking, the input has the needed information')
    
# -----------------------------------------------------------------------------
#  #function to trandform the initial input list in json
# -----------------------------------------------------------------------------

def input_transform (mylist):
    return { index : value for index,value in enumerate(mylist)}
