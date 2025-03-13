import wordninja
import pandas as pd

# Function to apply word splitting and cleaning operations
def split_words(df):
    # Ensure that 'title', 'content', 'summary' columns exist in the DataFrame
    text_columns = ['title', 'content', 'summary']
    
    for col in text_columns:
        # Apply word splitting using wordninja, handle missing/NaN values
        df[col] = df[col].fillna('').apply(lambda x: " ".join(wordninja.split(x)) if isinstance(x, str) else x)
        
        # Remove special characters and single characters
        df[col] = df[col].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters
        df[col] = df[col].str.replace(r'\b\w{1}\b', '', regex=True)  # Remove single character words
    
    return df

# Example: Assuming df is loaded from a previous step or function like load_data()
# Here we're just assuming the DataFrame is available.
if __name__ == "__main__":
    # You should load your data before processing
    # df = load_data() # Uncomment if you have load_data function
    
    # Clean text columns
    cleaned_df = split_words(df)
    
    # Display the cleaned DataFrame
    print(cleaned_df)

