# Definitions of the DHS Index dataset
######################################
DHS_INDEX_N_BIOSAMPLES = 733
DHS_INDEX_BIOSAMPLE_META_PATH = "data/dhs_index/raw/DHS_Index_and_Vocabulary_metadata.tsv"

# Definitions relevant to all models
####################################
MODEL_INPUT_LENGTH = 500

DATA_SPLITS_CHRS_PATH = "data/dhs_index/data_splits_chrs.json"
DATA_SPLITS_DHS_IDX_PATH = "data/dhs_index/data_splits_dhs_idx.h5"

# DHS64 definitions
###################
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

# DHS733 definitions
####################
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

# MPRA definitions
##################
MPRA_DATA_PATH = "data/mpra/enhancer_mpra_processed.tsv"
MPRA_DATA_SPLITS_IDX_PATH = "data/mpra/mpra_data_splits.json"
MPRA_CELL_LINES = [
    'NT2_D1', 'GM12878', '786_O', 'SKNSH', 'WERI_Rb1', 'SJCRH30', 'HepG2', 'K562', 'MCF7', 'HeLaS3', "HEK293", "HMC3",
]
MPRA_CELL_LINES_SORTED = [
    'HepG2', 'WERI_Rb1', 'SJCRH30', 'K562', 'HeLaS3', 'NT2_D1', 'GM12878', 'SKNSH', 'MCF7', '786_O', "HEK293", "HMC3",
]
MPRA_TARGETS = [
    'NT2_D1', 'GM12878', '786_O', 'SKNSH', 'WERI_Rb1', 'SJCRH30', 'HepG2', 'K562', 'MCF7', 'HeLaS3',
]
MPRA_TARGETS_SORTED = [
    'HepG2', 'WERI_Rb1', 'SJCRH30', 'K562', 'HeLaS3', 'NT2_D1', 'GM12878', 'SKNSH', 'MCF7', '786_O',
]
MPRA_CELL_LINES_EXTRA = ['HEK293', 'HMC3']
MPRA_TISSUES = [
    'Retina',
]
MPRA_TARGETS_DOUBLE_SORTED = [
    ('HepG2', 'K562'),
    ('HepG2', 'WERI_Rb1'),
    ('HeLaS3', 'HepG2'),
    ('GM12878', 'K562'),
    ('NT2_D1', 'SKNSH'),
    ('GM12878', 'SKNSH'),
    ('SJCRH30', 'SKNSH'),
    ('786_O', 'MCF7'),
]
MPRA_TARGETS_TRIPLE_SORTED = [
    ('GM12878', 'HepG2', 'NT2_D1'),
    ('GM12878', 'SJCRH30', 'WERI_Rb1'),
    ('HeLaS3', 'HepG2', 'MCF7'),
    ('HepG2', 'K562', 'SKNSH'),
    ('HeLaS3', 'K562', 'MCF7'),
    ('HeLaS3', 'NT2_D1', 'WERI_Rb1'),
    ('786_O', 'GM12878', 'SKNSH'),
    ('HeLaS3', 'SJCRH30', 'SKNSH'),
]

MPRA_CELL_LINES_LOG2FC_COLS = [f"log2FC_{ct}" for ct in MPRA_CELL_LINES]
MPRA_CELL_LINES_SORTED_LOG2FC_COLS = [f"log2FC_{ct}" for ct in MPRA_CELL_LINES_SORTED]
MPRA_TARGETS_LOG2FC_COLS = [f"log2FC_{ct}" for ct in MPRA_TARGETS]
MPRA_TARGETS_SORTED_LOG2FC_COLS = [f"log2FC_{ct}" for ct in MPRA_TARGETS_SORTED]

# DHS64-MPRA model definitions
##############################
DHS64_MPRA_MODEL_PATH = {
    0: "models/dhs64_mpra/dhs64_mpra_data_split_0.h5",
    1: "models/dhs64_mpra/dhs64_mpra_data_split_1.h5",
    3: "models/dhs64_mpra/dhs64_mpra_data_split_3.h5",
}
