DHS_INDEX_N_BIOSAMPLES = 733
DHS_INDEX_BIOSAMPLE_META_PATH = "data/dhs_index/raw/DHS_Index_and_Vocabulary_metadata.tsv"

MODEL_INPUT_LENGTH = 500

DATA_SPLITS_CHRS_PATH = "data/dhs_index/data_splits_chrs.json"
DATA_SPLITS_DHS_IDX_PATH = "data/dhs_index/data_splits_dhs_idx.h5"

DHS64_TRAIN_DATA_DIR = "data/dhs_index/dhs64_training"
DHS64_TRAIN_DATA_PATH = f"{DHS64_TRAIN_DATA_DIR}/dhs_metadata_logsignal_binary.csv.gz"
DHS64_TRAIN_META_PATH = f"{DHS64_TRAIN_DATA_DIR}/selected_biosample_metadata.tsv"

DHS64_BIOSAMPLE_META_PATH = "data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx"
DHS64_MODEL_PATH = {
    0: "models/dhs64/dhs64_data_split_0.h5",
    1: "models/dhs64/dhs64_data_split_1.h5",
    3: "models/dhs64/dhs64_data_split_3.h5",
}
DHS64_INPUT_LENGTH = MODEL_INPUT_LENGTH
DHS64_N_BIOSAMPLES = 64

DHS733_TRAIN_DATA_DIR = "data/dhs_index/dhs733_training"
DHS733_TRAIN_ONEHOT_SEQS_PATH = f"{DHS733_TRAIN_DATA_DIR}/dhs_seqs_onehot.h5"
DHS733_TRAIN_LOGSIGNAL_PATH = f"{DHS733_TRAIN_DATA_DIR}/dhs_metadata_logsignal.h5"

DHS733_BIOSAMPLE_META_PATH = DHS_INDEX_BIOSAMPLE_META_PATH
DHS733_MODEL_PATH = {
    0: "models/dhs733/dhs733_data_split_0.h5",
    1: "models/dhs733/dhs733_data_split_1.h5",
    3: "models/dhs733/dhs733_data_split_3.h5",
}
DHS733_INPUT_LENGTH = MODEL_INPUT_LENGTH
DHS733_N_BIOSAMPLES = 733
