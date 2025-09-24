import os
import sys
import requests

def ask_and_download(message, url_dest_list):
    reply = input(f"{message} (Y/N): ").strip().lower()
    if reply != 'y':
        return

    for url, dest_path in url_dest_list:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete.")
    print()

if __name__ == "__main__":

    # Trained DEN generator weights
    ###############################
    url_dest_list = [
        ("https://example-files.online-convert.com/document/txt/example.txt", "design/dhs64_single_den_pretrained_145nt"),
    ]
    ask_and_download("Download DHS64-trained DEN generator weights?", url_dest_list)
