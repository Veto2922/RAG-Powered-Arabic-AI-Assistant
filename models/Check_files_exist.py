import os
import gdown

class GoogleDriveDownloader:
    def __init__(self):
        self.sample_file_id = '1Og61VJzOMM8AJbhlsymWfJa3B9xxY-fW'
        self.persist_folder_url = 'https://drive.google.com/drive/folders/16sTPt066YDV22P55rh3awt2jQWZiOHLD'
        self.sample_directory = 'data/processed/all_split.pkl'
        self.persist_directory = 'data/processed/vector_db'
        self.prefix = 'https://drive.google.com/uc?/export=download&id='

    def download_file(self, file_id, destination):
        """Download a file from Google Drive if it doesn't exist or is 0 bytes."""
        if not os.path.exists(destination) or os.path.getsize(destination) == 0:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            print(f"The file {destination} does not exist or is 0 bytes. We will download it for you now, which may take some time.")
            print("Thank you for your patience!")
            gdown.download(self.prefix + file_id, destination, quiet=False)
        else:
            print(f"The file {destination} already exists and is valid.")

    def download_folder(self, url, destination):
        """Download a folder from Google Drive if it doesn't exist or is empty."""
        if not os.path.exists(destination) or not os.listdir(destination):
            print(f"The folder {destination} does not exist or is empty. We will download it for you now, which may take some time.")
            print("Thank you for your patience!")
            gdown.download_folder(url, quiet=False, output=destination)
        else:
            print(f"The folder {destination} already exists and is valid.")

    def download_sample_file(self):
        """Check and download the sample file."""
        if not os.path.exists(self.sample_directory) or os.path.getsize(self.sample_directory) == 0:
            print(f"{self.sample_directory} does not exist or is empty.")
            self.download_file(self.sample_file_id, self.sample_directory)
        else:
            print(f"{self.sample_directory} already exists and is valid.")

    def download_persist_folder(self):
        """Check and download the persist folder."""
        if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
            print(f"{self.persist_directory} does not exist or is empty. We will download it for you now, which may take some time.")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.download_folder(self.persist_folder_url, self.persist_directory)
        else:
            print(f"{self.persist_directory} already exists and is valid.")

# Example usage:
if __name__ == "__main__":
    # Define the directories and Google Drive file/folder IDs
   

    # Initialize the downloader class
    downloader = GoogleDriveDownloader()

    # Download the sample file and persist folder
    downloader.download_sample_file()
    downloader.download_persist_folder()
