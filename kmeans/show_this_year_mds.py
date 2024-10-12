import os
import datetime

def list_md_files(path):
    current_year = datetime.datetime.now().year
    md_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                # Get the modification time of the file
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                # Check if the file was modified this year
                if mod_time.year == current_year:
                    md_files.append(file_path)

    return md_files

# Usage example
path = '/Users/emacspy/Documents/_我的本地库思考'
md_files_this_year = list_md_files(path)

# Display the filtered files
for md_file in md_files_this_year:
    print(md_file)
