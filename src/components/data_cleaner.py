import pandas as pd

import numpy as np 

import warnings
warnings.filterwarnings('ignore')

import joblib
import cloudpickle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# 1. Function to clean the column names
def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #name mapper
    column_name_mapper = {'Temperature(°C)': 'Temperature', 'Humidity(%)': 'Humidity', 
                    'Wind speed (m/s)': 'Wind speed', 'Visibility (10m)': 'Visibility', 
                    'Dew point temperature(°C)': 'Dew point temperature', 'Solar Radiation (MJ/m2)': 'Solar Radiation', 
                    'Rainfall(mm)': 'Rainfall', 'Snowfall (cm)': 'Snowfall'
                    }
    
    try:
        df = df.rename(columns=column_name_mapper)  # rename
        df.columns = df.columns.str.lower() # lower case
        df.columns = df.columns.str.replace('\s+', '_', regex=True) # replace space with '_'
         
        df['date'] = pd.to_datetime(df['date'], dayfirst=True) # convert to datetime
        return df

    except Exception as E:
        print(f'\033[31m{type(E).__name__}: {E} !!!\033[0m')


class NullValueImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns
        self.cat_cols = X.select_dtypes(include=[object]).columns
        
        self.num_means = X[self.num_cols].mean()    # Calculate mean
        self.cat_modes = X[self.cat_cols].mode().iloc[0]    # Calculate Mode
        
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop_duplicates().reset_index(drop=True)  # Drop Duplicates

        # Transform null rows
        X[self.num_cols] = X[self.num_cols].fillna(self.num_means)
        X[self.cat_cols] = X[self.cat_cols].fillna(self.cat_modes)
        
        return X



if __name__ == "__main__":
    data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/SeoulBikeData.csv', 
                       encoding='unicode_escape')
    
    data = clean_col_names(data)
    
    NullImputer = NullValueImputer()
    NullImputer.fit_transform(data)

    print(data.shape)




