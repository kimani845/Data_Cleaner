
#this removes all the \n from the content table and summary
df["content"] = df["content"].str.replace("\n", " ", regex=True)
df["summary"] = df["summary"].str.replace("\n", " ", regex=True)
df["references"] = df["references"].str.replace("\n", " ", regex=True)
df["categories"] = df["categories"].str.replace("\n", " ", regex=True)


# Remove special characters and single characters
df['content'] = df['content'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters
df['summary'] = df['summary'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special 
df['references'] = df['references'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters
df['categories'] = df['categories'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters


df['content'] = df['content'].str.replace(r'\b\w{1}\b', '', regex=True)
df['summary'] = df['summary'].str.replace(r'\b\w{1}\b', '', regex=True) 
df['references'] = df['references'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters
df['categories'] = df['categories'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters

# Display the updated DataFrame
print(df)