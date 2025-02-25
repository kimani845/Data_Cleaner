def drop_duplicates():
    # Drop Duplicated values
    df = df.drop_duplicates()
# check for duplicates
print("\n **Duplicate Rows**")
print(f"The total number of duplicates is:{df.duplicated().sum()}")

