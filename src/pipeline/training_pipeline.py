import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.scripts.initiate_datamaker import DataSetSplitterInitiator
from src.scripts.initiate_preprocessor import PreProcessInitiator
from src.scripts.initiate_trainer import ModelTrainerInitiator

from src.utils import load_config, save_dataframe, save_pipeline_components


# Load the raw Data
raw_df = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/SeoulBikeData.csv', 
                     encoding='unicode_escape')

config = load_config("main_config.yaml")

model = RandomForestRegressor(n_estimators=250, 
                              max_depth=10, 
                              min_samples_split=13, 
                              random_state=42)

def training_pipeline(model=model, dataframe=raw_df):

    DataSplitterObj = DataSetSplitterInitiator(dataframe)
    train_df, val_df, test_df = DataSplitterObj.initiate_split()

    print(train_df.shape, val_df.shape, test_df.shape)

    PreProcessInitiatorObj = PreProcessInitiator()
    transformed_train_data = PreProcessInitiatorObj.fit_transform(train_df)
    transformed_val_data = PreProcessInitiatorObj.transform(val_df, data_type="val")
    transformed_test_data = PreProcessInitiatorObj.transform(test_df, data_type="test")

    print(transformed_train_data.shape, transformed_val_data.shape, transformed_test_data.shape)

    ModelTrainerInitiatorObj = ModelTrainerInitiator(model=model, 
                                                     train_df=transformed_train_data, 
                                                     val_df=transformed_val_data
                                                     )
    ModelTrainerInitiatorObj.initiate_training()

    print("Training Pipeline Completed")

    return 



if __name__ == "__main__":
    training_pipeline(model=model, dataframe=raw_df)