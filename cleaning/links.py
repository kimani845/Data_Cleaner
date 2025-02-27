
# Delete any links in the columns 'summary' and 'contents'
df['summary'] = df['summary'].str.replace(r'http[s]?://\S+', '', regex=True)
df['content'] = df['content'].str.replace(r'http[s]?://\S+', '', regex=True)
