# import pandas as pd
# import numpy as np
# import re

# def handle_missing_values(df, threshold=0.5):
#     """
#     Handles missing values in a dataset dynamically based on data type and threshold.
#     - Drops columns with missing values above threshold.
#     - Fills numeric columns using mean, median, or mode.
#     - Fills categorical columns using mode, forward fill, or backward fill.
#     - Uses interpolation for time-series data.
#     """
# def handle_missing_values():

#     # Drop columns with missing values above threshold
#     df = df.dropna(axis=1, thresh=int(threshold * len(df)))
    
#     for col in df.columns:
#         missing_count = df[col].isnull().sum()
#         if missing_count > 0:
#             print(f"Handling {missing_count} missing values in column: {col}")
            
#             if df[col].dtype in ['int64', 'float64']:  # Numeric columns
#                 if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
#                     df[col] = df[col].interpolate()  # Interpolation for time-series
#                 else:
#                     df[col].fillna(df[col].mean(), inplace=True)  # Default: Fill with mean
                    
#             elif df[col].dtype == 'object':  # Categorical columns
#                 if df[col].nunique() <= 5:  # Small unique values (likely categorical)
#                     df[col].fillna(df[col].mode()[0], inplace=True)
#                 else:
#                     df[col].fillna(method='ffill', inplace=True)  # Forward fill
            
#             else:
#                 df[col].fillna(df[col].mode()[0], inplace=True)  # Default mode fill
    
#     return df

# # Missing Values in
# missing_values = df.isnull()
# summary_percentage = missing_values['summary'].value_counts() / df['summary'].size 
# content_percentage = missing_values['content'].value_counts() / df['summary'].size 

# print(content_percentage)
# print(summary_percentage)


# # Example usage
# if __name__ == "__main__":
#     file_path = "your_dataset.csv"  # Update with your dataset file
#     df = read_file(file_path)
#     cleaned_df = handle_missing_values(df)
#     cleaned_df.to_csv("cleaned_dataset.csv", index=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#     print("Data cleaning complete. Saved as cleaned_dataset.csv") 

import pandas as pd
import numpy as np
import re
from cleaning.data_loader import load_data  # Import dataset loader

def handle_missing_values(df, threshold=0.5):
    """
    Handles missing values in a dataset dynamically based on data type and threshold.
    
    - Drops columns with missing values above the threshold.
    - Fills numeric columns using mean or interpolation if time-series.
    - Fills categorical columns using mode, forward fill, or backward fill.
    """
    
    if df is None or df.empty:
        print("Error: Dataframe is empty or not loaded.")
        return None

    # Drop columns with missing values above threshold
    df = df.dropna(axis=1, thresh=int(threshold * len(df)))
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"Handling {missing_count} missing values in column: {col}")
            
            if df[col].dtype in ['int64', 'float64']:  # Numeric columns
                if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
                    df[col] = df[col].interpolate()  # Interpolation for time-series
                else:
                    df[col].fillna(df[col].mean(), inplace=True)  # Default: Fill with mean
                    
            elif df[col].dtype == 'object':  # Categorical columns
                if df[col].nunique() <= 5:  # Small unique values (likely categorical)
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(method='ffill', inplace=True)  # Forward fill
            
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)  # Default mode fill
    
    return df

# Example usage
if __name__ == "__main__":
    df = load_data()  # Load dataset from dataloader.py

    if df is not None:
        cleaned_df = handle_missing_values(df)
        # No saving to file, just output the cleaned dataframe
        print("Data cleaning complete. You can now use the cleaned data.")
