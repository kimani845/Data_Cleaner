# # Import of necessary libraries
# import pandas as pd
# import numpy as np
# import re
# import nltk
# import os
# from collections import Counter
# import json


# import pandas as pd 
# import json
# import re
# import langdetect
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from langdetect import detect, DetectorFactory
# from langdetect.lang_detect_exception import LangDetectException


# # Class to handle dataset loading
# class Data:
#     # Function to load different types of datasets
#     def load_dataset(self, file_path):
#         """Loads the dataset based on its file type."""
#         try:
#             # Default file path if no other is provided
#             if file_path == "default":
#                 file_path = r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv"

#             # Check if file exists before processing
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"OOOOPS! Your file '{file_path}' does not exist.")

#             # Load different file types
#             if file_path.endswith('.csv'):
#                 return pd.read_csv(file_path)
#             elif file_path.endswith('.json'):
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                 return pd.DataFrame(data)
#             elif file_path.endswith('.txt'):
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     return pd.DataFrame({'Text': f.readlines()})  # Read lines into a DataFrame
#             elif file_path.endswith('.tsv'):
#                 return pd.read_csv(file_path, sep='\t', header=None)
#             elif file_path.endswith(('.xlsx', '.xls')):
#                 return pd.read_excel(file_path)
#             else:
#                 raise ValueError("SORRY! You Provided an Unsupported File Format.")

#         except Exception as e:
#             print(f"OOHH! NOO! There was an Error Loading the Dataset: {e}")
#             return None
#         try: 
#             # reasing a file with the correct encoding
#             return pd.read_csv(file_path, encoding='utf-8')
#         except UnicodeDecodeError:
#             print("UnicodeDecodeError encountered with UTF-8 encoding. Trying ISO-8859-1.")
#             return pd.read_csv(file_path, encoding='ISO-8859-1')
#         except Exception as e:
#             print(f"Error reading the file: {e}")
#             return None
# # Class to handle dataset loading   


# # Create an instance of Data and load the dataset
# data_loader = Data()
# df = data_loader.load_dataset("default")

# # Print dataset if loaded successfully
# if df is not None:
#     print(df)


# # Initialize the class
# finance = Data()
# file_path = "default"

# try:
#     df = finance.load_dataset(file_path)
#     print(df.head())
# except Exception as e:
#     print(e)

# # Display dataset info
# df.info()

# # display the datatypesof the columns
# print(df.dtypes)
# # datatypes for specicfic columns
# print(df['column_name'].dtype)

# # checking for mixed datatypes
# df.applymap(type).head()

# # to find different datatypes in all the columns
# type_counts = df.apply(type).nunique()
# print(type_counts)

# # Datatypes distribution in each colummn
# for col in df.columns:
#     print (f"Column: {col}")
#     print (df[col].apply(lambda x: type(x)).value_counts(), "\n")   



# # convert datatypes
# data =df
# print ("Original Datatypes")
# print(df.dtypes)
# # convert datatypes using the convert_dtypes
# new_df = df.convert_dtypes()
# print("\nNew datatypes ")
# print(new_df.dtypes)


# df["content"] = df["content"].str.replace(r"[^\w\s]", "", regex=True).str.replace("\n", "", regex=True)
# df["summary"] = df["summary"].str.replace(r"[^\w\s]", "", regex=True).str.replace("\n", "", regex=True)
# df["title"] = df["title"].str.replace(r"[^\w\s]", "", regex=True)

# df.head()

# # Convert the 'summary' and 'contents' column to lowercase
# df['summary'] = df['summary'].str.lower()
# df['content'] = df['content'].str.lower()

# # Display the updated DataFrame
# print(df)

# # check the length of the content and summary columns
# df['column'].value_counts()
# df['column'].value_counts()

# # Display the unique values in the 'column' column
# print(df['column'].unique())

# df.isna().sum()

# df.to_csv("kim.csv", index=False)





