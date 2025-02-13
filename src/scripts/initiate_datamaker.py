from src.components.dataset_maker import TrainValTestSplitter
from src.utils import load_config, save_dataframe


config = load_config("main_config.yaml")

class DataSetSplitterInitiator:
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.splitter = TrainValTestSplitter()

    def initiate_split(self):
        train_df, val_df, test_df = self.splitter.split_dataframe(self.dataframe)

        save_dataframe(train_df, config['raw_data']["train_data_path"]), 
        save_dataframe(val_df, config['raw_data']["val_data_path"]), 
        save_dataframe(test_df, config['raw_data']["test_data_path"])

        self.splitter.get_split_counts()

        return train_df, val_df, test_df 


# if __name__ == "__main__":
#     print(config) 

#     import pandas as pd

#     data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/SeoulBikeData.csv', 
#     encoding='unicode_escape')

#     DataSplitterObj = DataSetSplitterInitiator(data)
#     train_df, val_df, test_df = DataSplitterObj.initiate_split()

