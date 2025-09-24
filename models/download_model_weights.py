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

    # DHS64 model weights
    #####################
    url_dest_list = [
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64/dhs64_data_split_0.h5"),
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64/dhs64_data_split_1.h5"),
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64/dhs64_data_split_3.h5"),
    ]
    ask_and_download("Download DHS64 model weights?", url_dest_list)

    # DHS64 no-filter model weights
    ###############################
    url_dest_list = [
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64/dhs64_nofilt_data_split_3.h5"),
    ]
    ask_and_download("Download weights of DHS64 model trained on unfiltered DHS data?", url_dest_list)

    # DHS64-MPRA model weights
    ##########################
    url_dest_list = [
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64_mpra/dhs64_mpra_data_split_0.h5"),
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64_mpra/dhs64_mpra_data_split_1.h5"),
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs64_mpra/dhs64_mpra_data_split_3.h5"),
    ]
    ask_and_download("Download DHS64-MPRA model weights?", url_dest_list)

    # DHS733 model weights
    ######################
    url_dest_list = [
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs733/dhs733_data_split_0.h5"),
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs733/dhs733_data_split_1.h5"),
        ("https://example-files.online-convert.com/document/txt/example.txt", "dhs733/dhs733_data_split_3.h5"),
    ]
    ask_and_download("Download DHS733 model weights?", url_dest_list)
