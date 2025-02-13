import pandas as pd

from src.utils import load_config, save_dataframe, save_pipeline_components

from src.scripts.initiate_datamaker import DataSetSplitterInitiator
from src.scripts.initiate_preprocessor import PreProcessInitiator

# Load the raw Data
raw_df = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/SeoulBikeData.csv', 
                     encoding='unicode_escape')

config = load_config("main_config.yaml")


def training_pipeline(dataframe=raw_df):

    DataSplitterObj = DataSetSplitterInitiator(raw_df)
    train_df, val_df, test_df = DataSplitterObj.initiate_split()

    print(train_df.shape, val_df.shape, test_df.shape)

    PreProcessInitiatorObj = PreProcessInitiator()
    transformed_train_data = PreProcessInitiatorObj.fit_transform(train_df)
    transformed_val_data = PreProcessInitiatorObj.transform(val_df, data_type="val")
    transformed_test_data = PreProcessInitiatorObj.transform(test_df, data_type="test")

    print(transformed_train_data.shape, transformed_val_data.shape, transformed_test_data.shape)

training_pipeline()