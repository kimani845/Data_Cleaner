
# # Normalize the Dataframe
# text_cols = ['title','contents', 'summary','urls', 'links']
# for col in text_cols:
#     df[col] = df[col].astype(str).str.lower()

# # Remove numbers from the 'title', 'content', and 'summary' columns
# df['title'] = df['title'].apply(lambda x: remove_numbers(x))
# df['content'] = df['content'].apply(lambda x: remove_numbers(x))
# df['summary'] = df['summary'].apply(lambda x: remove_numbers(x))
