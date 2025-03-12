import os
import pandas as pd
import numpy as np

# Load the dataset
DATA_FOLDER = "data/raw"  # Path to the folder containing the dataset
# get the first file in the folder

def load_data(DATA_FOLDER):
    file_list = [f for f in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, f))]
    if len(file_list) == 0:
        print("No files found in the data folder.")
        exit()
    file_path = os.path.join(DATA_FOLDER, file_list[0])
    df = pd.read_csv(file_path) # Load the dataset
    print(f"Loaded dataset from {file_path}")
    return df, file_path

"""Another way to load data
DATA_FOLDER = "data/raw"  # Path to the folder containing the dataset

def load_data():
    # loads the first file in the folder
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"Data folder does not exist: {DATA_FOLDER}")
        
    # get the list of dataset files
    file_list = [f for f in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, f))]
    if len(file_list) == 0:
        raise FileNotFoundError(f"No files found in the data folder: {DATA_FOLDER}")

    file_path = os.path.join(DATA_FOLDER, file_list[0])
    df = pd.read_csv(file_path) # Load the dataset
    print(f"Loaded dataset from {file_path}")
    return df, file_path
"""



