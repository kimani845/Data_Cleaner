# Text normalization
# convert the dataset to lowercase
def text_lowercase(text):
    df['title'] = df['title'].apply(lambda x: x.lower())
    df['content'] = df['content'].apply(lambda x: x.lower())
    df['summary'] = df['summary'].apply(lambda x: x.lower())
    return df.head()

text_lowercase(text)
# another way to convert the dataset to lowercase is using the str.lower() method
def text_lowercase(text):
    return text.str.lower()
input_str = df['column', 'column', 'column']
text_lowercase(input_str)

