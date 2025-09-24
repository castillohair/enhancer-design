#!/usr/bin/env bash

# TODO: Replace with final URLs once we make the zenodo repo public

# Function that downloads a specified set of files from a list of (url, destination) pairs
download_files() {
    local files=("$@")
    for pair in "${files[@]}"; do
        url=$(echo $pair | awk '{print $1}')
        dest=$(echo $pair | awk '{print $2}')
        mkdir -p "$(dirname "$dest")"
        echo "Downloading $url to $dest..."
        curl -fsSL "$url" -o "$dest"
        if [[ $? -ne 0 ]]; then
            echo "Failed to download $url"
        fi
    done
}

# Processed DHS data files
##########################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt data/dhs_index/DHS_annotated_with_seqs_max_length_500.csv.gz"
    "https://example-files.online-convert.com/document/txt/example.txt data/dhs_index/data_splits_chrs.json"
    "https://example-files.online-convert.com/document/txt/example.txt data/dhs_index/data_splits_dhs_idx.h5"
    "https://example-files.online-convert.com/document/txt/example.txt data/dhs_index/dhs64_training/dhs_metadata_logsignal_binary.csv.gz"
    "https://example-files.online-convert.com/document/txt/example.txt data/dhs_index/dhs733_training/dhs_seqs_onehot.h5"
    "https://example-files.online-convert.com/document/txt/example.txt data/dhs_index/dhs733_training/dhs_metadata_logsignal.h5"
)

read -p "Download processed DHS data? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading processed DHS data files."
fi
echo ""

# Processed MPRA data
#####################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt data/mpra/enhancer_mpra_processed.tsv"
    "https://example-files.online-convert.com/document/txt/example.txt data/mpra/mpra_data_splits.json"
)

read -p "Download processed MPRA data? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading processed MPRA data files."
fi
echo ""

# DHS64 model weights
#####################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64/dhs64_data_split_0.h5"
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64/dhs64_data_split_1.h5"
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64/dhs64_data_split_3.h5"
)

read -p "Download DHS64 model weights? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading DHS64 model weights."
fi
echo ""

# DHS64 no-filter model weights
###############################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64/dhs64_nofilt_data_split_3.h5"
)
read -p "Download weights of DHS64 model trained on unfiltered DHS data? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading DHS64 no-filter model weights."
fi
echo ""

# DHS64-MPRA model weights
##########################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64_mpra/dhs64_mpra_data_split_0.h5"
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64_mpra/dhs64_mpra_data_split_1.h5"
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs64_mpra/dhs64_mpra_data_split_3.h5"
)
read -p "Download DHS64-MPRA model weights? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading DHS64-MPRA model weights."
fi
echo ""

# DHS733 model weights
######################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs733/dhs733_data_split_0.h5"
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs733/dhs733_data_split_1.h5"
    "https://example-files.online-convert.com/document/txt/example.txt models/dhs733/dhs733_data_split_3.h5"
)
read -p "Download DHS733 model weights? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading DHS733 model weights."
fi
echo ""

# Trained DEN generator weights
###############################
files_to_download=(
    "https://example-files.online-convert.com/document/txt/example.txt design/dhs64_single_den_pretrained_145nt"
)
read -p "Download weights of DEN generators specific to DHS64 biosamples? (Y/N): " confirm
if [[ "$confirm" == "Y" || "$confirm" == "y" ]]; then
    download_files "${files_to_download[@]}"
    echo "Done downloading DEN generator weights."
fi
echo ""
