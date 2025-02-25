
# word Splitting
import wordninja
import pandas as pd

df ['title'] = df['title'].apply(lambda x: " ".join(wordninja.split(x)))
df ['content'] = df['content'].apply(lambda x: " ".join(wordninja.split(x)))
df ['summary'] = df['summary'].apply(lambda x: " ".join(wordninja.split(x)))

# Remove special characters and single characters
df['content'] = df['content'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters
df['summary'] = df['summary'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)  # Remove special characters

df['content'] = df['content'].str.replace(r'\b\w{1}\b', '', regex=True)
df['summary'] = df['summary'].str.replace(r'\b\w{1}\b', '', regex=True) 

# Display the updated DataFrame
print(df)