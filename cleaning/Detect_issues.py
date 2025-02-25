# Detecting issues in the dataset.  
print("\n **Dataset Overview**")
df.info()

print("\n **Data type in column title**")
print(df['title'].dtype)

# to find different datatypes in all the columns
print("\n **Different Datatypes in all columns**")
type_counts = df.apply(type).nunique()
print(type_counts)
print("\n **Datatype Distribution in each column**")
# Datatypes distribution in each colummn
for col in df.columns:
    print (f"Column: {col}")
    print (df[col].apply(lambda x: type(x)).value_counts(), "\n")

# #check on missing value
missing_values = df.isnull()
missing_values.head()
#list of columns with missing values
for column in missing_values.columns.values.tolist():
    print(column)
    print (missing_values[column].value_counts())
    print("")
df.head()