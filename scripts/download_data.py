import gdown
import pyzipper
import os

def download_and_extract():
    file_id = '18coNcp0pMt_v-6YamN4di9RyMBu-H2-V'
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_destination = 'data.zip'
    extract_to = 'data'
    # The first Beijing Summer Olympics was in 2008
    password = b'2008'

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"Created directory: {extract_to}")

    print(f"Downloading file from Google Drive (ID: {file_id})...")
    try:
        gdown.download(url, zip_destination, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return

    print(f"Unzipping to {extract_to}...")
    try:
        with pyzipper.AESZipFile(zip_destination) as zf:
            zf.extractall(path=extract_to, pwd=password)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip: {e}")
    finally:
        if os.path.exists(zip_destination):
            os.remove(zip_destination)
            print(f"Removed temporary file: {zip_destination}")

if __name__ == "__main__":
    download_and_extract()
