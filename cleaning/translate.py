
# Translate a single column of the dataframe
async def translate_column(df, column_name):
    print(f"\nTranslating '{column_name}' to English...")
    
    # Collect texts from the column
    texts = df[column_name].tolist()
    
    # Batch translate the texts in parallel
    translated_texts = await batch_translate(texts)
    
    # Replace the original column with the translated texts
    df[column_name] = translated_texts

# Function to read the file with proper encoding
def read_file_with_encoding(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        print("UnicodeDecodeError encountered with UTF-8 encoding. Trying ISO-8859-1.")
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading the file: {e}")
        return ""

# Main asynchronous function
async def main():
    # Read the dataset (replace with your actual dataset)
    df = pd.read_csv(r"C:\Kimani\workspace_csv\Simba\CSVs\physics_data.csv")

    # List the columns (title, content, summary)
    print("Columns in the dataset:")
    print(df.columns.tolist())

    # Translate content for the columns: title, content, and summary
    for column in ['title', 'content', 'summary']:
        if column in df.columns:
            await translate_column(df, column)
    
    # Display the translated dataframe
    print("\nTranslated dataset:")
    print(df)

# If already inside a running event loop (e.g., in Jupyter or similar environments), use `await` directly
if __name__ == "__main__":
    try:
        # Check if the event loop is already running (use await directly in this case)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Running in an existing event loop, calling main() directly...")
            loop.create_task(main())
        else:
            # Otherwise, use asyncio.run() to start the event loop
            asyncio.run(main())
    except RuntimeError:
        # If no event loop is running, use asyncio.run()
        asyncio.run(main())