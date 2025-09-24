# Datasets

The datasets described below are necessary for different model training, sequence design, and data analysis programs included in this repository. The `download_data.py` script included in this folder can download each dataset individually.

## Data for training accessibility models

Processed data necessary for model training and some analysis will be downloaded into the [`dhs_index`](./dhs_index/) subfolder. The [`process_data.ipynb`](./dhs_index/process_data.ipynb) notebook reproduces our data processing workflow starting from raw published sources, but re-running this should not be necessary in most cases.

## Cell line and mouse retina MPRA results

Processed MPRA data will be downloaded into the [`mpra`](./mpra/) subfolder.

The downloaded file `enhancer_mpra_processed.tsv` contains annotations and processed MPRA measurements of tested enhancers. Annotation columns include:
- Sequence ID
- Enhancer sequence
- Source (i.e. genome, Fast SeqProp, DEN)
- Category (e.g. the value `single_fsp` corresponds to enhancers targing a single biosample designed with Fast SeqProp)
- Intended target biosample(s)
- Information about sequences designed via motif embedding (columns that start with `motif_`)
- Measured accessibility for genome-sourced sequences (columns starting with `dhs_signal_`)
- DHS64-predicted accessibilities (columns starting with `dhs_pred_`).

MPRA result columns include:
- Sequencing read counts in columns `{cell_type}_reads_r{i}`, where `i` corresponds to the replicate number.
- DESeq2 outputs for each assayed cell type, most importantly log2(RNA/DNA) estimates in columns `log2FC_{cell_type}`.

The downloaded file `mpra_data_splits.json` contains MPRA sequence IDs partitioned into 7 folds, used for finetuning the DHS64-MPRA model. The notebook [`make_data_splits.ipynb`](./mpra/make_data_splits.ipynb) can regenerate this file.

<!---
`.fastq` files containing raw sequencing data can be downloaded from GEO (TODO Add GEO link when available). However, only the processed data table is used in the code in this repository.
-->

<!---
## Processed TF motif database

Cooming soon!

## Precomputed TF motif alignments

Cooming soon!

## Precomputed enhancer and accessibility contributions

Coming soon!

## Other datasets used for analysis

### Human Protein Atlas Cancer Cell line dataset

Coming soon!

### GTEx tissue RNA-seq data

Coming soon!
-->