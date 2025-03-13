import os
import pandas as pd
import numpy as np
# Parent File
# it will be possible to run all the files from this one file folder
from cleaning.data_loader import load_data
from cleaning.Detect_issues import detect_issues
from cleaning.duplicated import drop_duplicates
from cleaning.links import remove_url
from cleaning.missing import handle_missing_values 
from cleaning.normalization import normalize_text
from cleaning.number_to_words import convert_to_words
# from cleaning.remove_numbers import remove_numbers
from cleaning.special_chars import clean_text_columns,  process_files
# from cleaning.Textualdata import remove_non_textual_data
# from cleaning.stopwords import remove_stopwords
from cleaning.tokenization import tokenize_text_columns, process_files
from cleaning.translate import translate_column, read_file_with_encoding, main
from cleaning.word_splitting import split_words
# from cleaning.whitespace import remove_whitespace
from structural_data_updated import load_and_clean_dataset
# from cleaning.links import is_valid_url

def main ():
    DATA_FOLDER = "data/raw"  # Path to the folder containing the dataset
    df = load_data(DATA_FOLDER)  # Load the raw dataset
    df

#     # This is the main function that will be run when the script is run
#     dataset = pd.read_csv(r"Default file path") # Load the dataset
# # perform the missing value handling
#     cleaned_dataset = handle_missing_values(dataset) # Clean the dataset
    
"""Folder paths"""
RAW_DATA_PATH = "data/raw"
CLEANED_DATA_PATH = "data/cleaned"
STRUCTURAL_DATA_PATH = "data/structured"

# # Ensure that the folder paths exist
# if not os.path.exists(RAW_DATA_PATH):
#     os.makedirs(RAW_DATA_PATH)  # Create the folder if it does not exist
#     if not os.path.exists(CLEANED_DATA_PATH):
#         os.makedirs(CLEANED_DATA_PATH)  # Create the folder if it does not exist
#         if not os.path.exists(STRUCTURTAL_DATA_PATH):
#             os.makedirs(STRUCTURTAL_DATA_PATH)  # Create the folder if it does not exist

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(CLEANED_DATA_PATH, exist_ok=True) 
os.makedirs(STRUCTURAL_DATA_PATH, exist_ok=True) 

# A function to process the files
def process_files():
    for file_name in os.listdir(RAW_DATA_PATH):
        file_path = os.path.join(RAW_DATA_PATH, file_name)   
        
        if file_name.endswith((".csv", ".txt", ".json")): #process only csv, txt, json, xlsx, xls files
            print(f"Processing {file_name}...")
            
            # loading the datasets
            if file_name.endswith(".csv"):
                dataset = pd.read_csv(file_path)
            elif file_name.endswith(".txt"):    
                dataset = pd.read_csv(file_path, sep="\t")
            elif file_name.endswith(".json"):
                    dataset = pd.read_json(file_path)
            elif file_name.endswith(".xlsx", ".xls"):
                    dataset = pd.read_excel(file_path)
            else:
                print("File format not supported")  
                continue  
            
                
            # dataset = load_and_clean_dataset(file_name)
            # dataset.to_csv(os.path.join(STRUCTURAL_DATA_PATH, file_name), index=False)
            # print(f"Done processing {file_name}...")
        
            # Cleaning the dataset
            cleaned_dataset = handle_missing_values(dataset)
            cleaned_dataset = detect_issues(cleaned_dataset)
            cleaned_dataset = drop_duplicates(cleaned_dataset)  
            cleaned_dataset = remove_url(cleaned_dataset)
            cleaned_dataset = normalize_text(cleaned_dataset)
            cleaned_dataset = number_to_words(cleaned_dataset)
            cleaned_dataset = remove_numbers(cleaned_dataset)
            cleaned_dataset = remove_special_characters(cleaned_dataset)
            cleaned_dataset = remove_non_textual_data(cleaned_dataset)
            cleaned_dataset = tokenize_text(cleaned_dataset)
            cleaned_dataset = translate_text(cleaned_dataset)
            cleaned_dataset = split_words(cleaned_dataset)
            # cleaned_dataset = remove_whitespace(cleaned_dataset)
            # cleaned_dataset = remove_stopwords(cleaned_dataset)
            # cleaned_dataset = is_valid_url(cleaned_dataset)
            # cleaned_dataset = remove_urls(cleaned_dataset)
            # cleaned_dataset = remove_punct(cleaned_dataset)
            
            # Save the cleaned dataset
            cleaned_file_name = f"cleaned_{file_name}"
            cleaned_file_path = os.path.join(CLEANED_DATA_PATH, cleaned_file_name)  
            cleaned_dataset.to_csv(cleaned_file_path, index=False)
            print(f"Done cleaning{file_name}...")
            
            #Structuring the cleaned data
            structured_dataset = load_and_clean_dataset(cleaned_dataset)
            structured_file_name = f"structured_{file_name}"    
            structured_file_path = os.path.join(STRUCTURAL_DATA_PATH, structured_file_name)
            structured_dataset.to_csv(structured_file_path, index=False)
            print(f"Done structuring {file_name}...")
            
            print("f Saved structured file as: {structured_file_path}")
            print("f Saved cleaned file as: {cleaned_file_path}")
            
if __name__ == "__main__":
    process_files()
    print("All files processed successfully")
# if __name__ == "__main__":    