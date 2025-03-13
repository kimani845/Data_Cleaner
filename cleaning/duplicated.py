# def getDuplicateColumns(df):
#     # Find duplicate columns that have identical data.
#     duplicateColumnNames = set()

#     for x in range(df.shape[1]):
#         col = df.iloc[:, x]

#         for y in range(x + 1, df.shape[1]):
#             otherCol = df.iloc[:, y]

#             # If columns have identical values, mark as duplicate
#             if col.equals(otherCol):
#                 duplicateColumnNames.add(df.columns[y])

#     return list(duplicateColumnNames)

# # Load dataset
# # df = greece.load_dataset(file_path)
# if __name__ == "__main__":
#     # Get the list of duplicate columns
#     duplicateColNames = getDuplicateColumns(df)
    
# if duplicateColNames:
#         print("Duplicate Columns:", duplicateColNames)

#         # Remove the duplicate columns
#         df = df.drop(columns=duplicateColNames)
#         print("\nDataFrame after removing duplicates:\n", df.head())
# else:
#     print("No duplicate columns found!")
    
# def drop_duplicates():
#     # Drop Duplicated values
#     df = df.drop_duplicates()
# # check for duplicates
# print("\n **Duplicate Rows**")
# print(f"The total number of duplicates is:{df.duplicated().sum()}")
import pandas as pd

def getDuplicateColumns(df):
    """Find and return duplicate columns with identical data."""
    duplicateColumnNames = set()

    for x in range(df.shape[1]):
        col = df.iloc[:, x]

        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]

            # If columns have identical values, mark as duplicate
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns[y])

    return list(duplicateColumnNames)

def drop_duplicates(df):
    """Drop duplicate rows from the dataset and return cleaned DataFrame."""
    return df.drop_duplicates()

if __name__ == "__main__":
    # Load dataset
    file_path = "your_dataset.csv"  # Replace with actual path
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        df = None  # Ensure df is always defined

    if df is not None:
        # Detect duplicate columns
        duplicateColNames = getDuplicateColumns(df)

        if duplicateColNames:
            print("Duplicate Columns:", duplicateColNames)

            # Remove duplicate columns
            df = df.drop(columns=duplicateColNames)
            print("\nDataFrame after removing duplicate columns:\n", df.head())
        else:
            print("No duplicate columns found!")

        # # Drop duplicate rows
        # df = drop_duplicates(df)

        # Check and print duplicate row count
        duplicate_count = df.duplicated().sum()
        print(f"\nTotal duplicate rows after cleaning: {duplicate_count}")
    else:
        print("No data to process.")
