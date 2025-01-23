import pandas as pd
import os

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


