# Parent File
# it will be possible to run all the files from this one file folder
from cleaning.missing import handle_missing_values # this import imports the handle_missing_values function from the missing.py file
from cleaning.duplicated import drop_duplicates


def main ():
    # This is the main function that will be run when the script is run
    dataset = pd.read_csv(r"Default file path") # Load the dataset
# perform the missing value habdling
    cleaned_dataset = handle_missing_values(dataset) # Clean the dataset
    
    # Text normalization
    normalized_dataset = normalize_text(cleaned_dataset) # Normalize the text in the dataset