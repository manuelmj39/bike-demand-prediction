import pandas as pd

from abc import ABC, abstractmethod

class DatasetSplitter(ABC):
    def __init__(self, train_split: float, val_split: float, test_split: float):
        pass

    @abstractmethod
    def split_dataframe():
        pass

    def get_split_counts():
        pass


class TrainValTestSplitter(DatasetSplitter):
    def __init__(self, train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1):
        # split percentages
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def split_dataframe(self, df: pd.DataFrame) -> (pd.DataFrame):
        #Splitting counts
        self.train_split_cnt = int(len(df) * self.train_split) 
        self.val_split_cnt = int(len(df) * self.val_split)  
        self.test_split_cnt = int(len(df) * self.test_split) 

        # Splitting Datasets
        train_df = df[:self.train_split_cnt]
        val_df = df[self.train_split_cnt:self.train_split_cnt + self.val_split_cnt].reset_index(drop=True)
        test_df = df[self.train_split_cnt + self.val_split_cnt: self.train_split_cnt + self.val_split_cnt + self.test_split_cnt].reset_index(drop=True)

        print("Splitted the Dataset Succesfully \n")
        return train_df, val_df, test_df
    
    def get_split_counts(self) -> None:
        print(f"Train set has {self.train_split_cnt}")
        print(f"Validation set has {self.val_split_cnt}")
        print(f"Test set has {self.test_split_cnt}")



if __name__ == "__main__":
    data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/SeoulBikeData.csv', 
                       encoding='unicode_escape')
    
    train_test_splitter = TrainValTestSplitter()
    train_df, val_df, test_df = train_test_splitter.split_dataframe(data)  
    train_test_splitter.get_split_counts()