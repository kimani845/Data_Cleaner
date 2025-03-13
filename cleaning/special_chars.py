import pandas as pd
import os
import re
from cleaning.data_loader import load_data  # Assuming load_data() is defined in dataloader.py

# Function to clean text columns dynamically
def clean_text_columns(df):
    # Identify columns with text data (string type)
    text_columns = df.select_dtypes(include=['object']).columns
    
    for column in text_columns:
        if column in df.columns:
            # Remove newline characters
            df[column] = df[column].str.replace("\n", " ", regex=True)
            # Remove special characters (keep only letters, numbers, and spaces)
            df[column] = df[column].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
            # Remove single characters (e.g., 'a', 'b', '1', etc.)
            df[column] = df[column].str.replace(r'\b\w{1}\b', '', regex=True)
    
    return df

# Function to process all files in the 'data/raw' directory
def process_files():
    # Load all datasets using the load_data() function from dataloader.py
    datasets = load_data()  # Assuming this returns a dictionary of {filename: dataframe}

    for df_name, df in datasets.items():  # Loop through each dataset
        # Clean text columns for each DataFrame
        cleaned_df = clean_text_columns(df)

        # Save the cleaned dataset to the 'data/cleaned' directory
        cleaned_file_path = os.path.join('data/cleaned', f"cleaned_{df_name}")
        cleaned_df.to_csv(cleaned_file_path, index=False)
        print(f"Data cleaning complete for {df_name}. Saved as {cleaned_file_path}")

# Run the processing function
if __name__ == "__main__":
    process_files()
