import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning) 
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error 
from sklearn.ensemble import *
from sklearn import ensemble
from sklearn.linear_model import *
import pickle
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
import gc



def mdape(y_true,y_pred, **kwargs):
    mdape = abs((y_pred-y_true)/y_true)
    return mdape
my_scorer = make_scorer(mdape, greater_is_better=False)


SEED = 2020

usecols = ['DateUpdateListing','Price','Latitude','Longitude','PostalCode',	
                'StateCode','City',	'Neighborhood',	'PropertyType',	'YearBuilt','Beds',	'Baths']


df= pd.read_csv(r'rent_new.csv', sep=';', usecols=usecols)

numcol = df.select_dtypes(include=[np.number]).columns.drop('Price')
catcol = df.select_dtypes(include=[np.object]).columns


def imputer(df):
    '''NaN values imputainon to mode for cats and 0 for nums'''
    df.Neighborhood = df.Neighborhood.fillna(df.City)
    df.YearBuilt = df.YearBuilt.replace(np.nan, df.YearBuilt.mode()[0]) 
    df.PropertyType = df.PropertyType.replace(np.nan, df.PropertyType.mode()[0]) 
    df = df.dropna(subset=['Price', 'Latitude', 'Longitude', 'StateCode', 'DateUpdateListing', 'City'])
    for col in df:
        if col in catcol:
            df.loc[:, col]=df.loc[:, col].fillna(df.loc[:, col].mode()[0])
        else: df.loc[:, col].fillna(0, inplace=True)
    return df

def encoder(df):

    '''value encoder by dictionaries'''
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

def feature_transformator(df):
    df.City = df.City+'_'+df.StateCode
    df['BedBath'] = df.Beds + df.Baths
    df.DateUpdateListing = pd.to_datetime(df.DateUpdateListing)
    df['YearRent'] = df['DateUpdateListing'].dt.year
    # df['MonthRent'] = df['DateUpdateListing'].dt.month

    return df

def drop_outliers(df):

    df = df[(df.Price<df.Price.quantile(0.96)) & (df.Price>df.Price.quantile(0.02))]
    df = df[((df.Longitude<-50)&(df.Longitude>-140))]
    df.Longitude = abs(df.Longitude)
    df = df[(df.Latitude<50)]
    df = df[df.BedBath<=16]
    df = df[df.YearRent>=2021]
    cols_to_drop = ['DateUpdateListing', 'YearRent', 'BedBath']
    df = df.drop(cols_to_drop, axis=1) 

    return df

def feature_extractor(df):
    df = imputer(df)
    df = feature_transformator(df)
    df = encoder(df)
    df = drop_outliers(df)
    df = df.fillna(-1)
    return df



def main(df):
    df = feature_extractor(df)
    X = df.drop('Price', axis=1).reset_index(drop=True)
    y = df.Price
    del df
    gc.collect()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
    # print(X_train.head(2))
    rf = RandomForestRegressor(max_depth=30, n_estimators=100, random_state=2020, n_jobs=-1)
    rf.fit(X_train, y_train)
    scores = (-1 * cross_val_score(rf, X_train, y_train,
                              cv=5,
                            scoring=my_scorer))
    score = mdape(y_test, rf.predict(X_test))
    print('Median Percentage Error: %.3f' % score.median())
    del X_train, X_test, y_train, y_test
    gc.collect()
    # save the model to disk
    filename = '../models/finalized_model.pkl'
    # joblib.dump(rf, open(filename,'wb'),  compress=3, protocol=-1)
    joblib.dump(rf, open(filename,'wb'),  compress=9, protocol=-1)

    return print ('Model has been saved!')

# model = model(df)
if __name__ == "__main__":
    main(df)