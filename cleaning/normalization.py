# # Text normalization
# # convert the dataset to lowercase
# def text_lowercase(text):
#     df['title'] = df['title'].apply(lambda x: x.lower())
#     df['content'] = df['content'].apply(lambda x: x.lower())
#     df['summary'] = df['summary'].apply(lambda x: x.lower())
#     return df.head()

# text_lowercase(text)
# # another way to convert the dataset to lowercase is using the str.lower() method
# def text_lowercase(text):
#     return text.str.lower()
# input_str = df['column', 'column', 'column']
# text_lowercase(input_str)

import pandas as pd
import os

# Text normalization function to convert the dataset to lowercase
def normalize_text(df):
    """
    Converts specified columns in the DataFrame to lowercase.
    """
    # Ensure the columns exist before applying the transformation
    columns_to_normalize = ['title', 'content', 'summary']
    
    for col in columns_to_normalize:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return df

# Process all CSV files in the 'data/raw' directory
def process_files_in_directory(directory):
    # Get a list of all CSV files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    for file in files:
        file_path = os.path.join(directory, file)
        print(f"Processing file: {file_path}")
        
        try:
            # Read the data
            df = pd.read_csv(file_path)
            
            # Apply text_lowercase function
            df = text_lowercase(df)
            
            # Optionally, save the cleaned file or just print it (not saving as per your request)
            # df.to_csv(f"data/cleaned/{file}", index=False)
            print(df.head())
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Example usage
if __name__ == "__main__":
    raw_data_directory = 'data/raw'  # Path to the raw data directory
    process_files_in_directory(raw_data_directory)
