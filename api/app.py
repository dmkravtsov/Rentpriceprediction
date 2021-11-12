import numpy as np
import math
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from loguru import logger  
import pandas as pd





def imputer(df):
    '''NaN values imputainon to mode for cats and 0 for nums'''
    df.Neighborhood = df.Neighborhood.fillna(df.City)
    df.City = df.City+'_'+df.StateCode
    df.YearBuilt = df.YearBuilt.replace([np.nan, '', ' ', '?'], 2021.0) 
    df.PropertyType = df.PropertyType.replace([np.nan, '', ' ', '?'], 'apartment') 
    df.Latitude = df.Latitude.replace([np.nan, '', ' ', '?'], 37.149616)
    df.Longitude = df.Longitude.replace([np.nan, '', ' ', '?'], 87.616904)
    df.Longitude = abs(df.Longitude)
    df.PostalCode = df.PostalCode.replace([np.nan, '', ' ', '?'], 60657)
    df.StateCode = df.StateCode.replace([np.nan, '', ' ', '?'], 'TX')
    df.Beds = df.Beds.replace([np.nan, '', ' ', '?'], 2)
    df.Baths = df.Baths.replace([np.nan, '', ' ', '?'], 2)

    return df

def encoder(df):

    state = pd.read_csv('StateCodeDict.csv', header=None, index_col=0, squeeze=True).to_dict()
    df.StateCode = df.StateCode.map(state)
    post = pd.read_csv('PostalCodeDict.csv', header=None, index_col=0, squeeze=True).to_dict()
    df.PostalCode = df.PostalCode.map(post)
    city = pd.read_csv('CityDict.csv', header=None, index_col=0, squeeze=True).to_dict()
    df.City = df.City.map(city)
    neighborhood = pd.read_csv('NeighborhoodDict.csv', header=None, index_col=0, squeeze=True).to_dict()
    df.Neighborhood = df.Neighborhood.map(neighborhood)
    prop = pd.read_csv('PropertyTypeDict.csv', header=None, index_col=0, squeeze=True).to_dict()
    df.PropertyType = df.PropertyType.map(prop)
    year = pd.read_csv('YearBuiltDict.csv', header=None, index_col=0, squeeze=True).to_dict()
    df.YearBuilt = df.YearBuilt.map(year)
    

    return(df)



def feature_extractor(df):
    df = imputer(df)
    df = encoder(df)
    df = df.fillna(-1)
    return df



    


# create instance of Flask app
app = Flask(__name__)
model = joblib.load(open('finalized_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 3)
    features = [x for x in request.form.values()]
    # features = request.get_json()
    df = pd.DataFrame([features], columns = ['Latitude', 'Longitude', 'PostalCode', 'StateCode',  'City',  'Neighborhood',  'PropertyType',  'YearBuilt',  'Beds',  'Baths'], dtype=float) 
    df = feature_extractor(df)
    prediction = model.predict(df)
    output = round(prediction[0])

    return render_template('index.html', prediction_text='Predicted price for this apartments ${}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)