import pandas as pd
import glob
import os

def convert_files_to_md():
    """
    Finds all .csv, .xlsx, and .xls files in the current directory,
    reads them into a pandas DataFrame, and saves them as
    Markdown table files.
    """
    
    # Find all files with the specified extensions
    # We create a combined list of all target files
    csv_files = glob.glob('*.csv')
    excel_files_xlsx = glob.glob('*.xlsx')
    excel_files_xls = glob.glob('*.xls')
    
    all_files = csv_files + excel_files_xlsx + excel_files_xls
    
    if not all_files:
        print("No .csv, .xlsx, or .xls files found in the current directory.")
        return

    print(f"Found {len(all_files)} files to convert...")

    for filepath in all_files:
        try:
            # Get the filename without the extension
            # os.path.splitext splits 'my_file.csv' into ('my_file', '.csv')
            base_filename = os.path.splitext(filepath)[0]
            extension = os.path.splitext(filepath)[1]
            
            output_filename = base_filename + '.md'
            
            df = None # Initialize DataFrame
            
            # Read the file based on its extension
            if extension == '.csv':
                df = pd.read_csv(filepath)
                print(f"Processing CSV: {filepath}")
            elif extension in ['.xlsx', '.xls']:
                # Read only the first sheet (sheet_name=0)
                df = pd.read_excel(filepath, sheet_name=0)
                print(f"Processing Excel: {filepath} (first sheet)")

            if df is not None:
                # Convert the DataFrame to a Markdown table string
                # index=False ensures we don't include the pandas row index
                # in the markdown table, which is usually desired.
                markdown_table = df.to_markdown(index=False)
                
                # Save the markdown string to the new .md file
                # Use encoding='utf-8' to support special characters
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(markdown_table)
                    
                print(f"Successfully converted {filepath} -> {output_filename}")
            
        except Exception as e:
            # Print an error message if anything goes wrong with a file
            print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    # This block runs when the script is executed directly
    convert_files_to_md()
    print("\nConversion process finished.")