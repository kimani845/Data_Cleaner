
# Delete any links in the columns 'column1' and 'column2'. this can go upto column3 and more.
df['column1'] = df['column1'].str.replace(r'http[s]?://\S+', '', regex=True)
df['column2'] = df['column2'].str.replace(r'http[s]?://\S+', '', regex=True)

# removing links from the summary and content columns
#remove urls
import string

def remove_url(text):
    if isinstance(text, str):
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

#This function removes punctuations
def remove_punct(text):
    if isinstance (text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return text

columns = ['column1', 'column2', 'column3', 'column3']

for col in columns:
    df[col] = df[col].astype(str).apply(lambda x: remove_punct(remove_url(x)))
