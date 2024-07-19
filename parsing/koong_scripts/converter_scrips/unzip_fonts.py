import os
import zipfile
from tqdm import tqdm

def unzip_all(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all zip files in the source folder
    zip_files = [f for f in os.listdir(source_folder) if f.endswith('.zip')]

    # Use tqdm to create a progress bar
    for zip_file in tqdm(zip_files, desc="Unzipping files"):
        zip_path = os.path.join(source_folder, zip_file)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create a subfolder for each zip file (optional)
                subfolder = os.path.join(destination_folder, os.path.splitext(zip_file)[0])
                os.makedirs(subfolder, exist_ok=True)
                
                # Extract the contents of the zip file
                zip_ref.extractall(subfolder)
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is not a valid zip file.")
        except Exception as e:
            print(f"Error extracting {zip_file}: {str(e)}")

# Usage
source_folder = '/Volumes/5tb_alex_drive/Scraped Fonts/fontmeme/downloaded_zips_fontmeme'
destination_folder = '/Volumes/5tb_alex_drive/Scraped Fonts/fontmeme/unzipped_fontmeme'

unzip_all(source_folder, destination_folder)