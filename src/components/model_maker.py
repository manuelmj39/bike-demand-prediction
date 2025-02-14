import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def availability_error_metric(actual_demand, predicted_demand, weights=None):
    """
    Calculate the Availability Error Metric (AEM) to ensure enough items are available on time.
    
    Parameters:
    - actual_demand: A list or numpy array of actual demand values.
    - predicted_demand: A list or numpy array of predicted demand values.
    - weights: Optional list or numpy array of weights for each time period/item.
    
    Returns:
    - AEM (float): The calculated Availability Error Metric.
    """

    # Ensure inputs are numpy arrays
    actual_demand = np.array(actual_demand)
    predicted_demand = np.array(predicted_demand)
    
    # Calculate the shortfall (only positive differences)
    shortfall = np.maximum(0, actual_demand - predicted_demand)
    
    # If weights are provided, apply them
    if weights is not None:
        weights = np.array(weights)
        weighted_shortfall = shortfall * weights
        aem = np.sum(weighted_shortfall)
    else:
        aem = np.sum(shortfall)
    
    total_demand = np.sum(actual_demand)

    return aem / total_demand


class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

        return self.model

    def evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        y_pred = y_pred.astype(int)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        aem = availability_error_metric(y, y_pred)

        metrics = {"mse": mse, 
                   "rmse": rmse,
                   "mae": mae,
                   "r2": r2, 
                   "aem": aem}
        
        return metrics

    def train_evaluate(self, X_train, y_train, X_val, y_val):
        self.train_model(X_train, y_train)
        train_metrics = self.evaluate_model(X_train, y_train)
        val_metrics = self.evaluate_model(X_val, y_val)
        
        return self.model, train_metrics, val_metrics

    def print_metrics(self, metrics):
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        print("\n")

    