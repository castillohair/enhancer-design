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

    # Processed DHS data files
    ##########################
    url_dest_list = [
        ("https://zenodo.org/records/17410822/files/DHS_annotated_with_seqs_max_length_500.csv.gz?download=1", "dhs_index/DHS_annotated_with_seqs_max_length_500.csv.gz"),
        ("https://zenodo.org/records/17410822/files/data_splits_chrs.json?download=1", "dhs_index/data_splits_chrs.json"),
        ("https://zenodo.org/records/17410822/files/data_splits_dhs_idx.h5?download=1", "dhs_index/data_splits_dhs_idx.h5"),
        ("https://zenodo.org/records/17410822/files/dhs_metadata_logsignal_binary.csv.gz?download=1", "dhs_index/dhs64_training/dhs_metadata_logsignal_binary.csv.gz"),
        ("https://zenodo.org/records/17410822/files/dhs_seqs_onehot.h5?download=1", "dhs_index/dhs733_training/dhs_seqs_onehot.h5"),
        ("https://zenodo.org/records/17410822/files/dhs_metadata_logsignal.h5?download=1", "dhs_index/dhs733_training/dhs_metadata_logsignal.h5"),
    ]
    ask_and_download("Download processed DHS data?", url_dest_list)

    # Processed MPRA data
    #####################
    url_dest_list = [
        ("https://zenodo.org/records/17410822/files/enhancer_mpra_processed.tsv?download=1", "mpra/enhancer_mpra_processed.tsv"),
        ("https://zenodo.org/records/17410822/files/mpra_data_splits.json?download=1", "mpra/mpra_data_splits.json"),
    ]
    ask_and_download("Download processed MPRA data?", url_dest_list)
