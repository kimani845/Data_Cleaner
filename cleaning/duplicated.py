def getDuplicateColumns(df):
    # Find duplicate columns that have identical data.
    duplicateColumnNames = set()

    for x in range(df.shape[1]):
        col = df.iloc[:, x]

        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]

            # If columns have identical values, mark as duplicate
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns[y])

    return list(duplicateColumnNames)

# Load dataset
# df = greece.load_dataset(file_path)
if __name__ == "__main__":
    # Get the list of duplicate columns
    duplicateColNames = getDuplicateColumns(df)
    
if duplicateColNames:
        print("Duplicate Columns:", duplicateColNames)

        # Remove the duplicate columns
        df = df.drop(columns=duplicateColNames)
        print("\nDataFrame after removing duplicates:\n", df.head())
else:
    print("No duplicate columns found!")
    
def drop_duplicates():
    # Drop Duplicated values
    df = df.drop_duplicates()
# check for duplicates
print("\n **Duplicate Rows**")
print(f"The total number of duplicates is:{df.duplicated().sum()}")
