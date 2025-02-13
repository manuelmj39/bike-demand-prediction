from src.components.data_cleaner import clean_col_names, NullValueImputer

from src.components.feature_extractor import SkewDiscretizer, CategoricalEncoder, LagFeatureCreator
from src.components.feature_extractor import extract_date_features, remove_multicollinear_features

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer 
from sklearn.base import BaseEstimator, TransformerMixin


class PreProcessPipeline:
    def __init__(self):
        pass

    def create_cleaning_pipeline(self):
        self.clean_col_transformer = FunctionTransformer(func=clean_col_names)
        self.null_value_imputer = NullValueImputer()

        self.cleaning_pipeline = Pipeline([
            ('clean_col_transformer', self.clean_col_transformer), 
            ('imputer', self.null_value_imputer)])
        
        return self.cleaning_pipeline
    
    def create_feature_pipeline(self):
        self.skew_discretizer = SkewDiscretizer()
        self.multicollinear_transformer = FunctionTransformer(func=remove_multicollinear_features)
        self.categorical_encoder = CategoricalEncoder()
        self.date_features_transformer = FunctionTransformer(func=extract_date_features)
        self.lag_features_transformer = LagFeatureCreator(lag_hours=24)

        self.feature_transformer_pipeline = Pipeline([
            ('skew_discretizer', self.skew_discretizer), 
            ('multicollinear_transformer', self.multicollinear_transformer), 
            ('categorical_encoder', self.categorical_encoder), 
            ('date_features_transformer', self.date_features_transformer), 
            ('lag_features_transformer', self.lag_features_transformer)
            ])
        
        return self.feature_transformer_pipeline
    
    def get_preprocessing_pipeline(self):
        self.create_cleaning_pipeline()
        self.create_feature_pipeline()

        self.preprocessing_pipeline = Pipeline([
            ('cleaning_pipeline', self.cleaning_pipeline), 
            ('feature_transform_pipeline', self.feature_transformer_pipeline)
            ])
        
        return self.preprocessing_pipeline
    

if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/train_data.csv')
    feature_pipe = PreProcessPipeline().get_preprocessing_pipeline()

    print(data.shape)
    print(data.columns)

    data = feature_pipe.fit_transform(data)

    print(data.shape)
    print(data.columns)