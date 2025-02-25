# Import of necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import os
from collections import Counter
import json


import pandas as pd 
import json
import re
import langdetect
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# Class to handle dataset loading
class Data:
    # Function to load different types of datasets
    def load_dataset(self, file_path):
        """Loads the dataset based on its file type."""
        try:
            # Default file path if no other is provided
            if file_path == "default":
                file_path = r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv"

            # Check if file exists before processing
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"OOOOPS! Your file '{file_path}' does not exist.")

            # Load different file types
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return pd.DataFrame({'Text': f.readlines()})  # Read lines into a DataFrame
            elif file_path.endswith('.tsv'):
                return pd.read_csv(file_path, sep='\t', header=None)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError("SORRY! You Provided an Unsupported File Format.")

        except Exception as e:
            print(f"OOHH! NOO! There was an Error Loading the Dataset: {e}")
            return None
        try: 
            # reasing a file with the correct encoding
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UnicodeDecodeError encountered with UTF-8 encoding. Trying ISO-8859-1.")
            return pd.read_csv(file_path, encoding='ISO-8859-1')
        except Exception as e:
            print(f"Error reading the file: {e}")
            return None
# Class to handle dataset loading   


# Create an instance of Data and load the dataset
data_loader = Data()
df = data_loader.load_dataset("default")

# Print dataset if loaded successfully
if df is not None:
    print(df)


# Initialize the class
finance = Data()
file_path = "default"

try:
    df = finance.load_dataset(file_path)
    print(df.head())
except Exception as e:
    print(e)

# Display dataset info
df.info()

# display the datatypesof the columns
print(df.dtypes)
# datatypes for specicfic columns
print(df['column_name'].dtype)

# checking for mixed datatypes
df.applymap(type).head()

# to find different datatypes in all the columns
type_counts = df.apply(type).nunique()
print(type_counts)

# Datatypes distribution in each colummn
for col in df.columns:
    print (f"Column: {col}")
    print (df[col].apply(lambda x: type(x)).value_counts(), "\n")   



# convert datatypes
data =df
print ("Original Datatypes")
print(df.dtypes)
# convert datatypes using the convert_dtypes
new_df = df.convert_dtypes()
print("\nNew datatypes ")
print(new_df.dtypes)


df["content"] = df["content"].str.replace(r"[^\w\s]", "", regex=True).str.replace("\n", "", regex=True)
df["summary"] = df["summary"].str.replace(r"[^\w\s]", "", regex=True).str.replace("\n", "", regex=True)
df["title"] = df["title"].str.replace(r"[^\w\s]", "", regex=True)

df.head()



# Drop rows where both columns 'content' and 'summary' are empty
df.dropna(subset=['content', 'summary'], how='all', inplace=True)

# Delete any links in the columns 'summary' and 'contents'
df['summary'] = df['summary'].str.replace(r'http[s]?://\S+', '', regex=True)
df['content'] = df['content'].str.replace(r'http[s]?://\S+', '', regex=True)

# Convert the 'summary' and 'contents' column to lowercase
df['summary'] = df['summary'].str.lower()
df['content'] = df['content'].str.lower()

# Display the updated DataFrame
print(df)

# check the length of the content and summary columns
df['column'].value_counts()
df['column'].value_counts()

# Display the unique values in the 'column' column
print(df['column'].unique())

df.isna().sum()

# Text normalization
# convert the dataset to lowercase
def text_lowercase(text):
    df['title'] = df['title'].apply(lambda x: x.lower())
    df['content'] = df['content'].apply(lambda x: x.lower())
    df['summary'] = df['summary'].apply(lambda x: x.lower())
    return df.head()

text_lowercase(text)
# another way to convert the dataset to lowercase is using the str.lower() method
def text_lowercase(text):
    return text.str.lower()
input_str = df['column', 'column', 'column']
text_lowercase(input_str)


# Convert the numbers to words with dictionary that has infinite numbers
# # Convert numbers to words  using a dictionary
# import re
# num_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

# tens_dict = {'10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', 
#                     '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty', '40': 'forty',
#                     '50': 'fifty', '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety'} 

# hundred_dict = {'100': 'one hundred', '200': 'two hundred', '300 ': 'three hundred', '400': 'four hundred', '500': 'five hundred',
#                     '600': 'six hundred', '700': 'seven hundred', '800': 'eight hundred', '900': 'nine hundred'} 
    
# Large_numbers = {
#     100: 'hundred', 1000: 'thousand', 1_000_000: 'million'
# }
# def convert_to_words(num):
    
#     num = int(num)  # Ensure input is an integer
#     if num < 10:
#         return num_dict[str(num)]
#     if num <20 :
#         return tens_dict[str(num)]
#     if num < 100:
#         tens, remainder = divmod(num, 10)
#         return tens_dict[str(tens * 10)] +(''if remainder == 0 else '-' + num_dict[str(remainder)])
#     if num < 1000:  
#         hundreds, remainder = divmod(num, 100)
#         return num_dict[str(hundreds)] + 'hundred' + ('' if remainder == 0 else '' + convert_to_words(remainder))
#     if num < 1_000_000:
#         thousands, remainder = divmod(num, 1000)
#         return convert_to_words(thousands) + 'thousand' + ('' if remainder == 0 else '' + convert_to_words(remainder))
        
#     for large_num in sorted(Large_numbers.keys(), reverse=True):
#         if num >= large_num:
#             large, remainder = divmod(num, large_num)
#             return convert_to_words(large) + ' ' + Large_numbers[large_num] + ('' if remainder == 0 else ' ' + convert_to_words(remainder))
#         # return num
        
#     for num in sorted(Large_numbers.keys(), reverse=True):
#         if num >= num:
#             large, remainder = divmod(num, num)
#             return convert_to_words(large) + ' ' + Large_numbers[large] + ('' if remainder == 0 else ' ' + convert_to_words(remainder))
#         return num
            
    
# def convert_numbers_to_words(text):

#     # text = text.str.replace(r'\d+', lambda x: num_dict[x.group()] 
#     #                         if len(x.group()) == 1 
#     #                         else tens_dict.get(x.group(), x.group()))
    
#     if isinstance(text, int): #if input is a number, convert it instantly
#         return convert_to_words(text)
#     if isinstance (text, str): # if the input is a text, replace the numbers that are within the string
#         text = re.sub(r'\d+', lambda x: convert_to_words(int(x.group())), text) 
#         return text 

#     return text    
# # text.apply(lambda x: convert_to_words(x))  
# # result = ' '.join([num_dict.get(i, i) for i in text.split()])
# # return result
    
# # Apply the function
# # Testing with a number
# input_number = 213423 
# print (convert_to_words(input_number))

# # testing with text containing numbers
# input_text = "I have 23 oranges and 4354 biscuits! Do you want some?"
# print (convert_to_words(input_text))

# Remove numbers
# Remove numbers from a string using regular expression
import re

num_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', 
            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

tens_dict = {'10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', 
             '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', 
             '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty', 
             '70': 'seventy', '80': 'eighty', '90': 'ninety'} 

hundred_dict = {'100': 'one hundred', '200': 'two hundred', '300': 'three hundred', '400': 'four hundred', 
                '500': 'five hundred', '600': 'six hundred', '700': 'seven hundred', '800': 'eight hundred', 
                '900': 'nine hundred'} 

Large_numbers = {100: 'hundred', 1000: 'thousand', 1_000_000: 'million'}

def convert_to_words(num):
    num = int(num)  # Ensure input is an integer

    if num < 10:
        return num_dict[str(num)]
    
    if num < 20:
        return tens_dict[str(num)]
    
    if num < 100:
        tens, remainder = divmod(num, 10)
        return tens_dict[str(tens * 10)] + ('' if remainder == 0 else '-' + num_dict[str(remainder)])
    
    if num < 1000:
        hundreds, remainder = divmod(num, 100)
        return num_dict[str(hundreds)] + ' hundred' + ('' if remainder == 0 else ' ' + convert_to_words(remainder))
    
    if num < 1_000_000:
        thousands, remainder = divmod(num, 1000)
        return convert_to_words(thousands) + ' thousand' + ('' if remainder == 0 else ' ' + convert_to_words(remainder))
    
    for large_num in sorted(Large_numbers.keys(), reverse=True):
        if num >= large_num:
            large, remainder = divmod(num, large_num)
            return convert_to_words(large) + ' ' + Large_numbers[large_num] + ('' if remainder == 0 else ' ' + convert_to_words(remainder))

def convert_numbers_to_words(text):
    if isinstance(text, int):  # If input is a number, convert it directly
        return convert_to_words(text)
    
    if isinstance(text, str):  # If input is text, replace numbers within the string
        text = re.sub(r'\d+', lambda x: convert_to_words(int(x.group())), text)
        return text
    
    return text



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

# Translate a single column of the dataframe
async def translate_column(df, column_name):
    print(f"\nTranslating '{column_name}' to English...")
    
    # Collect texts from the column
    texts = df[column_name].tolist()
    
    # Batch translate the texts in parallel
    translated_texts = await batch_translate(texts)
    
    # Replace the original column with the translated texts
    df[column_name] = translated_texts

# Function to read the file with proper encoding
def read_file_with_encoding(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        print("UnicodeDecodeError encountered with UTF-8 encoding. Trying ISO-8859-1.")
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading the file: {e}")
        return ""

# Main asynchronous function
async def main():
    # Read the dataset (replace with your actual dataset)
    df = pd.read_csv(r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv")

    # List the columns (title, content, summary)
    print("Columns in the dataset:")
    print(df.columns.tolist())

    # Translate content for the columns: title, content, and summary
    for column in ['title', 'content', 'summary']:
        if column in df.columns:
            await translate_column(df, column)
    
    # Display the translated dataframe
    print("\nTranslated dataset:")
    print(df)

# If already inside a running event loop (e.g., in Jupyter or similar environments), use `await` directly
if __name__ == "__main__":
    try:
        # Check if the event loop is already running (use await directly in this case)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Running in an existing event loop, calling main() directly...")
            loop.create_task(main())
        else:
            # Otherwise, use asyncio.run() to start the event loop
            asyncio.run(main())
    except RuntimeError:
        # If no event loop is running, use asyncio.run()
        asyncio.run(main())


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

            

# Apply the function
# Testing with a number
input_number = 213423
print(convert_numbers_to_words(input_number))

# Testing with text containing numbers
input_text = "I have 23 oranges and 4354 biscuits! Do you want some?"
print(convert_numbers_to_words(input_text))



import re
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result
input_str = df['column', 'column', 'column']
remove_numbers(input_str)

# Normalize the Dataframe

text_cols = ['title','contents', 'summary','urls', 'links']
for col in text_cols:
    df[col] = df[col].astype(str).str.lower()



df.to_csv("kim.csv", index=False)





