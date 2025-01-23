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


# Categorizing Skewed Columns
class SkewDiscretizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.discrete_rainfall(X)
        X = self.discrete_snowfall(X)
        X = self.discrete_visibility(X)
        X = self.discrete_radiation(X)
        return X

    @staticmethod
    def discrete_rainfall(df, col='rainfall'):
        # Define conditions for rainfall categories
        conditions = [
            df[col] == 0,
            (df[col] > 0) & (df[col] <= 3.5),
            (df[col] > 3.5)
            ]
        
        # Define corresponding category labels
        categories = ['No', 'Light', 'Medium']
        
        # Apply conditions to create the 'rainfall_class' column
        df['rainfall_class'] = np.select(conditions, categories, default='Unknown')
        df.drop([col], axis=1, inplace=True)
        
        return df
    
    @staticmethod   
    def discrete_snowfall(df, col='snowfall'):
        # Define conditions for snowfall categories
        conditions = [
            df[col] == 0,
            (df[col] > 0) & (df[col] <= 0.5),
            (df[col] > 0.5) & (df[col] <= 2.0),
            (df[col] > 2.0) & (df[col] <= 4.0),
            df[col] > 4.0
        ]
        
        # Define corresponding category labels
        categories = ['No', 'Light', 'Medium', 'Heavy', 'Extreme']
        
        # Apply conditions to create the 'snowfall_class' column
        df['snowfall_class'] = np.select(conditions, categories, default='Unknown')
        df.drop([col], axis=1, inplace=True)

        return df
    
    @staticmethod
    def discrete_visibility(df, col='visibility'):    
        df[f'{col}_scaled'] = (df[col] * 10) / 1000

        # Define conditions for visibility categories
        conditions = [
            (df[f'{col}_scaled'] <= 5),
            (df[f'{col}_scaled'] > 5) & (df[f'{col}_scaled'] <= 10),
            df[f'{col}_scaled'] > 10
        ]

        # Define corresponding category labels
        categories = ['Poor', 'Moderate', 'Good']
        
        # Apply conditions to create the 'visibility_class' column
        df['visibility_class'] = np.select(conditions, categories, default='Unknown')
        df.drop([col], axis=1, inplace=True)

        return df.drop([f'{col}_scaled'], axis=1)

    @staticmethod
    def discrete_radiation(df, col='solar_radiation'):
        # Define conditions for visibility categories
        conditions = [
            df[col] <= 0.5,
            (df[col] > 0.5) & (df[col] <= 1),
            (df[col] > 1) & (df[col] <= 2.5),
            (df[col] > 2.5) & (df[col] <= 5),
            df[col] > 5
        ]

        # Define corresponding category labels
        categories = ['Very Low', 'Low', 'Moderate', 'High', 'Extreme']
        
        # Apply conditions to create the 'solar_radiation_class' column
        df['solar_radiation_class'] = np.select(conditions, categories, default='Unknown')
        df.drop([col], axis=1, inplace=True)
        return df
    

# Remove higly correlated features
def remove_multicollinear_features(df):
    features_to_remove = ['dew_point_temperature']

    return df.drop(features_to_remove, axis=1)


# Encode Categorical Features
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=[object]).columns
        self.cat_cols_mapping = {'seasons': {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}, 
                                 'holiday': {'No Holiday': 0, 'Holiday': 1},
                                 'functioning_day': {'No': 0, 'Yes': 1},
                                 'rainfall_class': {'No': 0, 'Light': 1, 'Medium': 2},
                                 'snowfall_class': {'No': 0, 'Light': 1, 'Medium': 2, 'Heavy': 3, 'Extreme': 4},
                                 'visibility_class': {'Poor': 0, 'Moderate': 1, 'Good': 2},
                                 'solar_radiation_class': {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Extreme': 4}
        }
        
        return self

    def transform(self, X):
        
        X = X.copy()
        
        for col in self.cat_cols:
            X[col] = X[col].map(self.cat_cols_mapping[col])
        
        return X
    
# Extract Time Features
def extract_date_features(df):
    """
    Extracts date features from the 'date' column of the dataframe.

    Args:
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    df : pd.DataFrame
        DataFrame with extracted date features.
    """

    df = df.copy()
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    df = df.drop(['date'], axis=1)
    return df


# Create lag features
class LagFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, lag_hours=1):
        self.lag_hours = lag_hours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for i in range(1, self.lag_hours + 1):
            X[f'lag_{i}'] = X['rented_bike_count'].shift(i)

        X = X.dropna().reset_index(drop=True)
        return X
