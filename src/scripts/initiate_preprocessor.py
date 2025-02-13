from src.utils import load_config, save_dataframe, save_pipeline_components
from src.components.preprocessing_pipeline import PreProcessPipeline


config = load_config("main_config.yaml")

class PreProcessInitiator:
    def __init__(self):
        self.PreProcessPipelineObj = PreProcessPipeline()
        self.pipeline = self.PreProcessPipelineObj.get_preprocessing_pipeline()

    def fit_transform(self, dataframe):
        transformed_dataframe = self.pipeline.fit_transform(dataframe)

        # Save the pipeline
        save_pipeline_components(self.PreProcessPipelineObj,
                                 config['components']["transformer_component_path"],  
                                 config['components']["pipeline_component_path"])

        save_dataframe(transformed_dataframe, config['transformed_data']["train_data_path"])

        return transformed_dataframe
    
    def transform(self, dataframe, data_type="val"):
        transformed_dataframe = self.pipeline.transform(dataframe)

        if data_type == "val":
            save_dataframe(transformed_dataframe, config['transformed_data']["val_data_path"])

        elif data_type == "test":
            save_dataframe(transformed_dataframe, config['transformed_data']["test_data_path"])

        return transformed_dataframe
    
# if __name__ == "__main__":
#     import pandas as pd

#     train_data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/train_data.csv')
#     val_data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/validation_data.csv')
#     test_data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/raw_data/test_data.csv')

#     PreProcessInitiatorObj = PreProcessInitiator()
#     transformed_train_data = PreProcessInitiatorObj.fit_transform(train_data)
#     transformed_val_data = PreProcessInitiatorObj.transform(val_data, data_type="val")
#     transformed_test_data = PreProcessInitiatorObj.transform(test_data, data_type="test")




    

