# # Detect issues in the dataset
# import pandas as pd
# import os
# import pandas as pd
# from langdetect import detect
# import asyncio
# import pandas as pd
# from googletrans import Translator
# import re
# from cleaning.data_loader import load_data


# def detect_issues(df):
#     # checks if the dataset has some common issues and report them
#        print("\n **Dataset Overview**")
#        print(df.info())
    
#        print("\n **Missing Values")
#        print(df.isnull().sum())
    
#        print("\n **Duplicate Rows**")
#        print(f"The total number of duplicates is:{df.duplicated().sum()}")
    
#        print("\n **Inconsistent Datatypes**")
#        print(df.apply(type).head())


# detect_issues(df)
# print("detect_issues")

# # #check on missing value
# missing_values = df.isnull()
# missing_values.head()
# #list of columns with missing values
# for column in missing_values.columns.values.tolist():
#     print(column)
#     print (missing_values[column].value_counts())
#     print("")
# df.head()

# # DETECT LANGUAGE
# # Function to detect if the language is not English
# def detect_non_english(text):
#     """Detects if the language is not English ('en')."""
#     try:
#         # Check if text is a string and detect the language
#         if isinstance(text, str):
#             return detect(text) != 'en'  # Return True if the text is not English
#         return False  # If the text is not a string (e.g., NaN), return False
#     except Exception:
#         return False  # In case of error, assume it's English

# # # Load dataset 
# # file_path = r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv"
# # df = pd.read_csv(file_path)

# # Column in the dataset
# columns_to_check = []

# # Ensure all columns exist
# for column in columns_to_check:
#     if column not in df.columns:
#         print(f"Column '{column}' not found in the dataset!")
#         columns_to_check.remove(column)

# # Check each column for non-English text
# for column in columns_to_check:
#     # Apply the detection function to each of the columns (title, content, summary)
#     df[f'{column}_is_non_english'] = df[column].apply(detect_non_english)

# # Filter and print rows where any column has non-English text
# non_english_rows = df[df[['title_is_non_english', 'content_is_non_english', 'summary_is_non_english']].any(axis=1)]

# # Print the rows that contain non-English text in any of the specified columns
# print(non_english_rows[['title', 'content', 'summary'] + [f'{col}_is_non_english' for col in columns_to_check]])


# # Batch translation function
# async def batch_translate(texts, dest='en'):
#     """Batch translate a list of texts."""
#     translator = Translator()
#     try:
#         translations = await asyncio.gather(*[translator.translate(text, dest=dest) for text in texts])
#         return [translation.text for translation in translations]
#     except Exception as e:
#         print(f"Error during batch translation: {e}")
#         return texts
    

# # Function to detect outliers
# def detect_outliers(df):
#     print("\nChecking for the outliers in the Dataset:")
    
#     for col in df.select_dtypes(include=['number']).columns:  
#         # Detecting outliers based on 5th and 95th percentiles
#         outliers = df[(df[col] < df[col].quantile(0.05)) | (df[col] > df[col].quantile(0.95))]
#         print(f"Outliers in '{col}': {len(outliers)}")

# # Function to validate email format
# def check_email_format(email):
#     """Simple regex-based email format checker."""
#     email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
#     return re.match(email_regex, email) is not None

# # Function to check for incorrect email format in the dataset
# def check_email(df):
#     print("\n **Incorrect Email Format**")
    
#     # Search for a column containing 'email' in its name
#     if any(df.columns.str.contains("email", case=False)):
#         email_col = df.loc[:, df.columns.str.contains("email", case=False)].columns[0]
        
#         # Check for invalid emails in the column
#         invalid_emails = df[~df[email_col].astype(str).apply(check_email_format)]
#         print(f"Number of emails with incorrect format in '{email_col}': {len(invalid_emails)}")
#     else:
#         print("No email column found.")

# # df = pd.read_csv(file_path) # Load the dataset
# detect_outliers(df)
# check_email(df)


import pandas as pd
import os
import re
import asyncio
from langdetect import detect
from googletrans import Translator
from cleaning.data_loader import load_data  # Ensure this module exists

# Load dataset dynamically
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Detect common dataset issues
def detect_issues(df):
    print("\n** Dataset Overview **")
    print(df.info())

    print("\n** Missing Values **")
    print(df.isnull().sum())

    print("\n** Duplicate Rows **")
    print(f"Total duplicates: {df.duplicated().sum()}")

    print("\n** Inconsistent Data Types **")
    print(df.dtypes)

# Detect missing values
def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("\n** Missing Values **")
    print(missing_values[missing_values > 0])

# Detect language of text columns
def detect_non_english(text):
    try:
        if isinstance(text, str):
            return detect(text) != 'en'
        return False
    except Exception:
        return False  # Assume English in case of error

def check_non_english_text(df, text_columns):
    for col in text_columns:
        if col in df.columns:
            df[f'{col}_is_non_english'] = df[col].apply(detect_non_english)
    
    non_english_rows = df[df[[f"{col}_is_non_english" for col in text_columns]].any(axis=1)]
    print("\n** Non-English Text Rows **")
    print(non_english_rows[text_columns + [f"{col}_is_non_english" for col in text_columns]])

# Async batch translation
async def batch_translate(texts, dest='en'):
    translator = Translator()
    try:
        translations = await asyncio.gather(*[asyncio.to_thread(translator.translate, text, dest=dest) for text in texts])
        return [translation.text for translation in translations]
    except Exception as e:
        print(f"Error during batch translation: {e}")
        return texts

# Detect outliers
def detect_outliers(df):
    print("\n** Checking for Outliers **")
    for col in df.select_dtypes(include=['number']).columns:
        outliers = df[(df[col] < df[col].quantile(0.05)) | (df[col] > df[col].quantile(0.95))]
        print(f"Outliers in '{col}': {len(outliers)}")

# Validate email format
def check_email_format(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(email_regex, email))

# Check for incorrect email formats
def check_email(df):
    email_cols = [col for col in df.columns if "email" in col.lower()]
    
    if not email_cols:
        print("\n** No Email Column Found **")
        return

    print("\n** Checking Email Formats **")
    for email_col in email_cols:
        invalid_emails = df[~df[email_col].astype(str).apply(check_email_format)]
        print(f"Incorrect emails in '{email_col}': {len(invalid_emails)}")

# Main function
if __name__ == "__main__":
    file_path = "data/sample.csv"  # Change this to the actual dataset path
    df = load_dataset(file_path)

    if df is not None:
        detect_issues(df)
        check_missing_values(df)
        detect_outliers(df)
        check_email(df)
        
        # Specify columns to check for non-English text
        text_columns = ["title", "content", "summary"]
        check_non_english_text(df, text_columns)
