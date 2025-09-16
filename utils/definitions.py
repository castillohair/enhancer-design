DHS_INDEX_N_BIOSAMPLES = 733

DATA_SPLITS_CHRS_PATH = "data/dhs_index/data_splits_chrs.json"
DATA_SPLITS_DHS_IDX_PATH = "data/dhs_index/data_splits_dhs_idx.h5"

DHS64_TRAIN_DATA_DIR = "data/dhs_index/dhs64_training"
DHS64_TRAIN_DATA_PATH = f"{DHS64_TRAIN_DATA_DIR}/dhs_metadata_logsignal_binary.csv.gz"
DHS64_TRAIN_META_PATH = f"{DHS64_TRAIN_DATA_DIR}/selected_biosample_metadata.tsv"

DHS733_TRAIN_DATA_DIR = "data/dhs_index/dhs733_training"
DHS733_TRAIN_ONEHOT_SEQS_PATH = f"{DHS733_TRAIN_DATA_DIR}/dhs_seqs_onehot.h5"
DHS733_TRAIN_LOGSIGNAL_PATH = f"{DHS733_TRAIN_DATA_DIR}/dhs_metadata_logsignal.h5"

MODEL_MAX_SEQ_LEN = 500