import os
import sys
import requests
import zipfile

def ask_and_download(message, url_dest_list):
    reply = input(f"{message} (Y/N): ").strip().lower()
    if reply != 'y':
        return

    for url, dest_path in url_dest_list:
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete.")

if __name__ == "__main__":

    # Trained DEN generator weights
    ###############################
    url_dest_list = [
        ("https://zenodo.org/records/17410822/files/dhs64_single_den_pretrained_145nt.zip?download=1", "dhs64_single_den_pretrained_145nt.zip"),
    ]
    ask_and_download("Download DHS64-trained DEN generator weights?", url_dest_list)
    # check if the file was downloaded
    # if so, create a folder and unzip the file there
    if os.path.exists("dhs64_single_den_pretrained_145nt.zip"):
        print("Unzipping DHS64-trained DEN generator weights...")
        with zipfile.ZipFile("dhs64_single_den_pretrained_145nt.zip", 'r') as zip_ref:
            zip_ref.extractall("dhs64_single_den_pretrained_145nt")
        print("Unzipped DHS64-trained DEN generator weights.")
        # remove the zip file
        os.remove("dhs64_single_den_pretrained_145nt.zip")
        print("Done.")
