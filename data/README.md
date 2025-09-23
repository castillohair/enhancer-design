# Datasets

The following datasets are used in this repository:

## Data for training accessibility models

Located in the [`dhs_index`](./dhs_index/) subfolder. Processed data necessary for model training and some analysis can be downloaded via the included [`download_processed_data.sh`](./dhs_index/download_processed_data.sh) script. The [`process_data.ipynb`](./dhs_index/process_data.ipynb) jupyter notebook reproduces our data processing workflow starting from raw, previously-published sources, but re-running this should not be necessary in most cases.

## Cell line and mouse retina MPRA results

Located in the [`mpra`](./mpra/) subfolder. The script [`download_mpra_results.sh`](./dhs_index/download_processed_data.sh) downloads a `.tsv` file containing annotations and processed MPRA measurements of tested enhancers that passed sequencing quality filters. Annotation columns include:
- Sequence id
- Enhancer sequence
- Source (i.e. genome, Fast SeqProp, DEN)
- Category (e.g. the value `single_fsp` corresponds to enhancers targing a single biosample designed with Fast SeqProp), intended target biosample(s)
- Information about motif embedding (columns that start with `motif_`)
- Measured accessibility for genome-sourced sequences (columns starting with `dhs_signal_`)
- DHS64-predicted accessibilities (columns starting with `dhs_pred_`).

MPRA result columns include
- Sequencing read counts (`{cell_type}_reads_r{i}` where `i` corresponds to the replicate number)
- DESeq2 outputs for each assayed cell type, most importantly log2(RNA/DNA) estimates (`log2FC_{cell_type}`).

`.fastq` files containing raw sequencing data can be downloaded from GEO (TODO Add GEO link when available). However, only the processed data table is used in the code in this repository.

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