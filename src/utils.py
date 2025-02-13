import pandas as pd
import os

import yaml

import cloudpickle
from sklearn.pipeline import Pipeline

#folder to load config file
CONFIG_PATH = r"config/"

# Function to load yaml configuration file
def load_config(config_name: str) -> dict:
    """
    Load configuration file

    This function load the configuration file which
    consists of all the paths, and other miscellaneous
    contents.

    Parameters
    ----------
        config_path: str
            Path to the configuration file

    Returns
    -------
        dict
            Dictionary of configurations
    """
    with open(os.path.join(CONFIG_PATH, config_name)) as config_file:
        config = yaml.safe_load(config_file)

    return config


# Function to save DataFrame with error handling
def save_dataframe(df: pd.DataFrame, path: str) -> None:
    try:
        # Check if the input is a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The provided input is not a valid pandas DataFrame.")
        
        # Check if the directory exists; create it if it doesn't
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the DataFrame
        df.to_csv(path, index=False)
        print(f"DataFrame successfully saved to {path}")
    
    except FileNotFoundError:
        print(f"Error: The specified path '{path}' is invalid.")
    except PermissionError:
        print(f"Error: Permission denied when trying to save to '{path}'.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return

# Function to save Pipeline Components
def save_pipeline_components(pipeline_obj, transformer_root, pipeline_root):
    transformer_directory = os.path.dirname(transformer_root)
    pipeline_directory = os.path.dirname(pipeline_root)

    os.makedirs(transformer_directory, exist_ok=True), os.makedirs(pipeline_directory, exist_ok=True)

    for key in pipeline_obj.__dict__.keys():
        if type(pipeline_obj.__dict__[key]) != Pipeline:
            with open(f"{transformer_root}/{key}.pkl", 'wb') as f:
                cloudpickle.dump(pipeline_obj.__dict__[key], f)

        else:
            with open(f"{pipeline_root}/{key}.pkl", 'wb') as f:
                cloudpickle.dump(pipeline_obj.__dict__[key], f)

    return

