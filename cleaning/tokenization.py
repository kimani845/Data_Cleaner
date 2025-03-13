import pandas as pd
import os

# Assuming load_data is already implemented in dataloader.py
from cleaning.data_loader import load_data

# Function to perform word tokenization on text columns
def tokenize_text_columns(df):
    # Automatically identify columns with string data type (i.e., text columns)
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        # Tokenize text by splitting on whitespace and create a new column for tokens
        df[f'{col}_tokens'] = df[col].astype(str).apply(lambda x: x.split())

    return df

# Function to process all files in the 'data/raw' directory
def process_files():
    # Load all datasets using the load_data() function from dataloader.py
    datasets = load_data()  # Assuming this returns a dictionary of {filename: dataframe}

    for df_name, df in datasets.items():  # Loop through each dataset
        # Perform word tokenization for each DataFrame
        tokenized_df = tokenize_text_columns(df)

        # Save the tokenized dataset to the 'data/cleaned' directory
        cleaned_file_path = os.path.join('data/cleaned', f"tokenized_{df_name}")
        tokenized_df.to_csv(cleaned_file_path, index=False)
        print(f"Tokenization complete for {df_name}. Saved as {cleaned_file_path}")

# Run the processing function
if __name__ == "__main__":
    process_files()
