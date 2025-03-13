
# # Delete any links in the columns 'column1' and 'column2'. this can go upto column3 and more.
# df['column1'] = df['column1'].str.replace(r'http[s]?://\S+', '', regex=True)
# df['column2'] = df['column2'].str.replace(r'http[s]?://\S+', '', regex=True)

# # removing links from the summary and content columns
# #remove urls
# import string

# def remove_url(text):
#     if isinstance(text, str):
#         return re.sub(r'https?://\S+|www\.\S+', '', text)
#     return text

# #This function removes punctuations
# def remove_punct(text):
#     if isinstance (text, str):
#         return text.translate(str.maketrans('', '', string.punctuation))
#     return text

# columns = ['column1', 'column2', 'column3', 'column3']

# for col in columns:
#     df[col] = df[col].astype(str).apply(lambda x: remove_punct(remove_url(x)))
# import pandas as pd
# import re
# import string

# # Function to remove URLs from text
# def remove_url(text):
#     if isinstance(text, str):
#         return re.sub(r'https?://\S+|www\.\S+', '', text)
#     return text

# # Function to remove punctuations
# def remove_punct(text):
#     if isinstance(text, str):
#         return text.translate(str.maketrans('', '', string.punctuation))
#     return text

# # List of columns to process
# columns = ['column1', 'column2', 'column3']  # Removed duplicate column3

# # Ensure all columns exist in df
# columns = [col for col in columns if col in df.columns]

# # Apply transformations
# for col in columns:
#     df[col] = df[col].astype(str).apply(lambda x: remove_punct(remove_url(x)) if pd.notna(x) else x)

# print("URLs and punctuation removed successfully.")
import re
import string
import pandas as pd

def remove_url(text):
    """Removes URLs from a given text."""
    if isinstance(text, str):
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

def remove_punct(text):
    """Removes punctuation from a given text."""
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return text

def clean_text_columns(df, columns_to_clean=None):
    """
    Cleans specified text columns in a dataframe by removing URLs and punctuation.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns_to_clean (list): List of column names to clean. If None, all text columns are cleaned.

    Returns:
    pd.DataFrame: The cleaned dataframe.
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or not loaded.")
        return None

    # If no specific columns are given, clean all object (string) columns
    if columns_to_clean is None:
        columns_to_clean = df.select_dtypes(include=['object']).columns.tolist()

    # Ensure only valid columns are processed
    columns_to_clean = [col for col in columns_to_clean if col in df.columns]

    if not columns_to_clean:
        print("No valid text columns found for cleaning.")
        return df

    for col in columns_to_clean:
        df[col] = df[col].astype(str).apply(lambda x: remove_punct(remove_url(x)) if pd.notna(x) else x)

    print("URLs and punctuation removed successfully from:", columns_to_clean)
    return df
