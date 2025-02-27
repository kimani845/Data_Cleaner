# Detect issues in the dataset
def detect_issues(df):
    # checks if the dataset has some common issues and report them
       print("\n **Dataset Overview**")
       print(df.info())
    
       print("\n **Missing Values")
       print(df.isnull().sum())
    
       print("\n **Duplicate Rows**")
       print(f"The total number of duplicates is:{df.duplicated().sum()}")
    
       print("\n **Inconsistent Datatypes**")
       print(df.apply(type).head())
    
df = pd.read_csv(r"Default file path")

detect_issues(df)
print("detect_issues")
# print (detect_issues(df))


# #check on missing value
missing_values = df.isnull()
missing_values.head()
#list of columns with missing values
for column in missing_values.columns.values.tolist():
    print(column)
    print (missing_values[column].value_counts())
    print("")
df.head()

# DETECT LANGUAGE
import pandas as pd
from langdetect import detect

# Function to detect if the language is not English
def detect_non_english(text):
    """Detects if the language is not English ('en')."""
    try:
        # Check if text is a string and detect the language
        if isinstance(text, str):
            return detect(text) != 'en'  # Return True if the text is not English
        return False  # If the text is not a string (e.g., NaN), return False
    except Exception:
        return False  # In case of error, assume it's English

# Load dataset 
file_path = r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv"
df = pd.read_csv(file_path)

# Column in the dataset
columns_to_check = ['title', 'content', 'summary']

# Ensure all columns exist
for column in columns_to_check:
    if column not in df.columns:
        print(f"Column '{column}' not found in the dataset!")
        columns_to_check.remove(column)

# Check each column for non-English text
for column in columns_to_check:
    # Apply the detection function to each of the columns (title, content, summary)
    df[f'{column}_is_non_english'] = df[column].apply(detect_non_english)

# Filter and print rows where any column has non-English text
non_english_rows = df[df[['title_is_non_english', 'content_is_non_english', 'summary_is_non_english']].any(axis=1)]

# Print the rows that contain non-English text in any of the specified columns
print(non_english_rows[['title', 'content', 'summary'] + [f'{col}_is_non_english' for col in columns_to_check]])

import asyncio
import pandas as pd
from googletrans import Translator

# Batch translation function
async def batch_translate(texts, dest='en'):
    """Batch translate a list of texts."""
    translator = Translator()
    try:
        translations = await asyncio.gather(*[translator.translate(text, dest=dest) for text in texts])
        return [translation.text for translation in translations]
    except Exception as e:
        print(f"Error during batch translation: {e}")
        return texts
    
    
import pandas as pd
import re

# Function to detect outliers
def detect_outliers(df):
    print("\nChecking for the outliers in the Dataset:")
    
    for col in df.select_dtypes(include=['number']).columns:  
        # Detecting outliers based on 5th and 95th percentiles
        outliers = df[(df[col] < df[col].quantile(0.05)) | (df[col] > df[col].quantile(0.95))]
        print(f"Outliers in '{col}': {len(outliers)}")

# Function to validate email format
def check_email_format(email):
    """Simple regex-based email format checker."""
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

# Function to check for incorrect email format in the dataset
def check_email(df):
    print("\n **Incorrect Email Format**")
    
    # Search for a column containing 'email' in its name
    if any(df.columns.str.contains("email", case=False)):
        email_col = df.loc[:, df.columns.str.contains("email", case=False)].columns[0]
        
        # Check for invalid emails in the column
        invalid_emails = df[~df[email_col].astype(str).apply(check_email_format)]
        print(f"Number of emails with incorrect format in '{email_col}': {len(invalid_emails)}")
    else:
        print("No email column found.")

df = pd.read_csv(r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv")  # Load your dataset
detect_outliers(df)
check_email(df)

            