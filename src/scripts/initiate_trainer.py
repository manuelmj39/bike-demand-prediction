from sklearn.ensemble import RandomForestRegressor

from src.components.model_maker import ModelTrainer
from src.utils import XySplitter, save_model, load_config


config = load_config("main_config.yaml")

class ModelTrainerInitiator:
    def __init__(self, model, train_df, val_df):
        self.model = model

        self.X_train, self.y_train = XySplitter(train_df).split()
        self.X_val, self.y_val = XySplitter(val_df).split()

    def initiate_training(self):
        model_trainer = ModelTrainer(self.model)
        self.model, train_metrics, val_metrics = model_trainer.train_evaluate(self.X_train, self.y_train, self.X_val, self.y_val)

        model_trainer.print_metrics(train_metrics)
        model_trainer.print_metrics(val_metrics)

        save_model(self.model, config['model']["model_path"] + config['model']["model_name"])

        return self.model
    
if __name__ == "__main__":
    import pandas as pd

    train_data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/transformed_data/train_data.csv')
    val_data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/transformed_data/validation_data.csv')
    test_data = pd.read_csv('/Users/manueljohn/Training/github-projects/bike-demand-prediction/artifacts/transformed_data/test_data.csv')

    ModelTrainerInitiatorObj = ModelTrainerInitiator(model=RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_split=13,
        random_state=42
    )
    , train_df=train_data, val_df=val_data)

    ModelTrainerInitiatorObj.initiate_training()


