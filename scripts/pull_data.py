#File for downloading data from google drive link

FOLDER_LINK = "https://drive.google.com/drive/u/1/folders/1LnxZkuHUxu0fw093sZlqC333_O0c3flt"

METADATA_LINK = "https://drive.google.com/file/d/1Hv4IiE6VKURiRXIeyjQjQtj2rDEAUG9u/view?usp=sharing"

import gdown
import os
from pathlib import Path


# Extract folder ID from Google Drive link
FOLDER_ID = "1LnxZkuHUxu0fw093sZlqC333_O0c3flt"

# Extract file ID from Google Drive link
METADATA_FILE_ID = "1Hv4IiE6VKURiRXIeyjQjQtj2rDEAUG9u"

# Download entire folder
PROJECT_ROOT = Path(__file__).parent.parent
output_path = PROJECT_ROOT / "data" / "activations_data"
output_path.mkdir(parents=True, exist_ok=True)

# Check if directory already has files
existing_files = list(output_path.glob("*"))
if existing_files:
    print(f"Found {len(existing_files)} existing files/folders in {output_path}")
    print("Skipping download. Delete the directory to re-download.")
else:
    print("Downloading entire folder from Google Drive...")
    gdown.download_folder(f"https://drive.google.com/drive/folders/{FOLDER_ID}", output=str(output_path), quiet=False, use_cookies=False)


