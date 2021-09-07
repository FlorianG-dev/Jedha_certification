# -----------------------------------------------------------------------------
#  We import what we need
# -----------------------------------------------------------------------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve
import statsmodels.api as st
from sklearn.preprocessing import PolynomialFeatures



import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
# setting Jedha color palette as default
pio.templates["jedha"] = go.layout.Template(
    layout_colorway=["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"]
)
pio.templates.default = "jedha"
pio.renderers.default = "notebook" # to be replaced by "iframe" if working on JULIE

# -----------------------------------------------------------------------------
#  # Visualize ROC curves
# -----------------------------------------------------------------------------


def plot_roc(X_train, Y_train, X_test, Y_test, classifier):
    
    probas_train = classifier.predict_proba(X_train)[:,1]
    precisions, recalls, thresholds = roc_curve(Y_train, probas_train)
    fig = go.Figure(
        data = go.Scatter(
            name = 'train',
            x = recalls, 
            y = precisions, 
            mode = 'lines'
        ),
        layout = go.Layout(
            title = go.layout.Title(text = "ROC curve", x = 0.5),
            xaxis = go.layout.XAxis(title = 'False Positive Rate'),
            yaxis = go.layout.YAxis(title = 'True Positive Rate')
        )
    )

    probas_test = classifier.predict_proba(X_test)[:,1]
    precisions, recalls, thresholds = roc_curve(Y_test, probas_test)
    fig.add_trace(go.Scatter(
        name = 'test',
        x = recalls, 
        y = precisions, 
        mode = 'lines'
        )
    )
    fig.show()


# -----------------------------------------------------------------------------
#  function of the machine 
#  The goal of this function is to process the data 
# -----------------------------------------------------------------------------


def machine (X,Y, model):
    
    #Divide categorical and numerical
    print("Encoding categorical features and standardizing numerical features...")
    categorical_features = list(X.select_dtypes(exclude=['int64','float64']).columns)
    numeric_features = list(X.select_dtypes(include=['int64','float64']).columns)
    print('  Categorical features:', *categorical_features)
    print('  Numeric features:', *numeric_features)
    print("...Done.")
    print('---')
    
    # Divide dataset Train set & Test set 
    print("Dividing into train and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify = Y)
    print("...Done.")
    print('---')
    
    # the preprocessings for X_train and X_test
    print("Encoding categorical features and standardizing numerical features...")

    numeric_transformer = StandardScaler()
    categorical_transformer =  OneHotEncoder(drop='first')
    categorical_transformer.fit(X_train[categorical_features]) #We fit the model one time to get the parameters letter
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print("...Done")
    print('---')
    
    print('features names after first transfo:', categorical_transformer.get_feature_names())
    print('---')
    col_list=categorical_transformer.get_feature_names()
    
    #We add a polynomial transformation
    print('We add a polynomial transformation:')
    poly = PolynomialFeatures(interaction_only=True)
    X_train =poly.fit_transform(X_train)
    print("...Done")
    print('---')
    X_test = poly.transform(X_test)
 
    #We rename the cols with their right names
    dic = {'0': 'Age', '1':'New_user', '2' : 'Germany', '3' : 'UK', '4': 'US', '5' : 'Direct', '6' : 'SEO'}
    poly_col = poly.get_feature_names()
    for i, v in enumerate(poly_col):
        for key,value in dic.items():
                poly_col[i] = poly_col[i].replace(key, value)
    print('the new features : ')
    print(poly_col)
    
    # We fit the model on our train set
    print("Training model...")
    fitted_model = model.fit(X_train, Y_train) # Training is always done on train set !!
    print("...Done.")

    # Predictions on training set
    print("Predictions on training set...")
    Y_train_pred = model.predict(X_train)
    print("...Done.")

    # Predictions on test set
    print("Predictions on test set...")
    Y_test_pred = model.predict(X_test)
    print("...Done.")
    print('---')

    # Print scores
    print("f1-score on train set : ", f1_score(Y_train, Y_train_pred))
    print("f1-score on test set : ", f1_score(Y_test, Y_test_pred))
    print('---')
    
    # Print the parameters when gridsearch
    try:
        print('\nThe best parameters are:\n')
        print(model.best_params_)
    except:
        pass
    
    # Print the features importances
    try:
        print()
        print('\nThe feature importances are:\n')
        print(model.feature_importances_)
    except:
        pass
    
    print('The heatmap :')
    plot_heat (Y_train,Y_train_pred, Y_test, Y_test_pred)
    print('The roc :')
    plot_roc (X_train, Y_train, X_test, Y_test, model)
    
    # Some additional stats using the OLS model : 
    print('---')
    print('\n additional stats:')
    X_train = st.add_constant(X_train)
    X_test = st.add_constant(X_test)
    ols = st.OLS(Y_train, X_train)
    ols_fit = ols.fit()
    display(ols_fit.summary())
    print('---')
    print('Columns coeff and pvalues for OLS model')
    
    
    pd_stats = pd.DataFrame(index = poly_col[1:], data = {'pvalues' : ols_fit.pvalues.values[1:] } ).sort_values("pvalues")
    display(pd_stats)
    
    return(fitted_model)


# -----------------------------------------------------------------------------
#  function to plot a heatmap
# -----------------------------------------------------------------------------


def plot_heat (Y_train,Y_train_pred, Y_test, Y_test_pred):
    cm_train = confusion_matrix(Y_train, Y_train_pred, normalize = 'true')
    cm_test = confusion_matrix(Y_test, Y_test_pred, normalize = 'true')
    
    plt.figure(figsize=(20,5))

    plt.subplot(121) 
    sns.heatmap(cm_train, annot= True, cmap= 'coolwarm').set_title('Train')
    plt.subplot(122) 
    sns.heatmap(cm_test, annot= True, cmap= 'coolwarm').set_title('Test')
    plt.show();