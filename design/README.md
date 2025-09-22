# Model-based enhancer design scripts

## Description and general usage

Scripts that reproduce the enhancer design tasks in the article are included here. These tasks include:

- Design enhancers specific to one biosample (i.e. cell type), across all [biosamples modeled by DHS64](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx): [`dhs64_single_fsp.py`](#dhs64_single_fsppy) (uses Fast SeqProp) [`dhs64_single_den.py`](#dhs64_single_denpy) (uses DENs).
- Design enhancers specific to multiple DHS64-modeled biosamples: [`dhs64_multiple_fsp.py`](#dhs64_multiple_fsppy).
- Design enhancers specific to one DHS64-modeled biosample with tunable target activity: [`dhs64_single_tunable.py`](#dhs64_single_tunable_fsppy).
- Design enhancers specific to any biosample in the [DNase I Index](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2559-3/MediaObjects/41586_2020_2559_MOESM3_ESM.xlsx): [`dhs733_single_fsp.py`](#dhs733_single_fsppy).

Scripts use [Fast SeqProp](https://doi.org/10.1186/s12859-021-04437-5) or [Deep Exploration Networks (DENs)](https://doi.org/10.1016/j.cels.2020.05.007)

### Command-line interface
Scripts can be used via their command-line interface with a variation of the following:

```shell
python {design_script}.py --target-idx {i} --n-seqs 100 --seq-length 145
```

Where `{i}` corresponds to the index of the biosample to target, i.e. for which the designed enhancers will be specific. See each individual subsection below for each script's precise interface.

### Outputs

The output of each script is a directory with the following files:

- `{output-prefix}_4mer_distance.png`: Violin plot of the normalized euclidean distance between 4mer counts across generated sequences, as a measure of sequence diversity.
- `{output-prefix}_editdistance.png`: Violin plot of the normalized edit distance between generated sequences, as a measure of sequence diversity.
- `{output-prefix}_preds_design_boxplot.png`: Box plot of accessibilities of generated sequences as predicted by the model used during design. This model is always an ensemble of two DHS64 or DHS733 models independently trained on distinct data splits.
- `{output-prefix}_preds_design.csv.gz`: Table with accessibilities of generated sequences as predicted by the design model.
- `{output-prefix}_preds_val_boxplot.png`: Box plot of accessibilities of generated sequences as predicted by a validation model, trained on a distinct data split than the models used for design.
- `{output-prefix}_preds_val.csv.gz`: Table with accessibilities of generated sequences as predicted by the validation model.
- `{output-prefix}_run_metadata.json`: Information about parameters used for sequence generation, including arguments provided to Fast SeqProp and DEN.
- `{output-prefix}_seq_bitmap.png`: Bitmap representing the designed sequences, where rows correspond to individual sequences and pixels represent nucleotide identity (i.e. A, C, G, or T).
- `{output-prefix}_seqs.fasta`: Generated sequences in .fasta format.
- `{output-prefix}_seqs.png`: Logos of the first few sequences.
- `{output-prefix}_train_history.png`: Plots of the different training losses across iterations. For the meaning of these, see the Fast SeqProp and DEN publications and their github repositories.

`output-prefix` can be specified via a command-line argument, otherwise it will default to `{target_idx}_{target_biosample_name}`. Some scripts create additional files, as indicated in each subsection below.

### Script code organization

These scripts can also be used as starting points to code custom sequence design tasks. Code within the scripts is roughly organized as follows:

1. Imports and constants: Import necessary libraries and define paths to model files and metadata.
2. Loss functions: These specify sequence features to optimize. Features include scores computed from predicted accessibilites (e.g. the difference between target and non-target predictions which we would like to maximize), as well as sequence features that we want to penalize such as long nucleotide repeats.
3. Main sequence design function: Runs Fast SeqProp or trains DENs to generate sequences, makes and saves predictions using both the design and validation models, and creates various plots to analyze the generated sequences.
4. Entry point: Parses command-line arguments and runs the main design function.

Note that most optimization-related arguments provided to Fast SeqProp and DEN are not exposed to the command-line API. Thus, if needed, these have to be changed by modifying their values in the code.

## `dhs64_single_fsp.py`

Design enhancers for activity specific to one of the [DHS64-modeled biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) against all others. Uses Fast SeqProp. This script can be run with the following arguments:

```
Run Fast SeqProp to generate sequences with biosample-specific activity using DHS64.

optional arguments:
  -h, --help            show this help message and exit
  --target-idx TARGET_IDX
                        Target biosample index within all DHS64-modeled biosamples.
  --n-seqs N_SEQS       Number of sequences to generate.
  --seq-length SEQ_LENGTH
                        Length of sequences to generate.
  --output-dir OUTPUT_DIR
                        Directory to save output files.
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files. If None, a prefix based on biosample index and name will be used.
  --seed SEED           Random seed for sequence initialization. If None, a random seed will be used.
```

To reproduce the design of enhancers in the article, run the following:

```shell
python dhs64_single_fsp.py --target-idx {i} --n-seqs 250 --seq-length 145
```

for every value of `{i}` between 0 and 63 inclusive.

## `dhs64_single_den.py`

Train a DEN to generate enhancers with activity specific to one of the [DHS64-modeled biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) against all others. This script also allows to use a previously trained DEN to generate new sequences.

Note that the required file `dhs64_single_den_generator.py` contains definitions related to the generator network. Currently the network architecture only allows for 145nt-long sequences, and should be modified if a different length is needed.

This script can be run with the following arguments:

```
Train a Deep Exploration Network to generate sequences with biosample-specific activity using DHS64.

optional arguments:
  -h, --help            show this help message and exit
  --target-idx TARGET_IDX
                        Target biosample index within all DHS64-modeled biosamples.
  --n-seqs N_SEQS       Number of sequences to generate.
  --output-dir OUTPUT_DIR
                        Directory to save output files.
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files. If None, a prefix based on biosample index and name will be used.
  --generator-path GENERATOR_PATH
                        Path to a pre-trained generator model. If None, a new generator will be trained.
  --random-seed-train RANDOM_SEED_TRAIN
                        Random seed for generator training.
  --random-seed-gen RANDOM_SEED_GEN
                        Random seed for sequence generation.
```

This script generates the following additional output files:
- `{output-prefix}_generator.h5`: Saved weights of the trained generator.

To reproduce the design of enhancers in the article, run the following:

```shell
python dhs64_single_den.py --target-idx {i} --n-seqs 250
```

for every value of `{i}` between 0 and 63 inclusive.

```
TODO: add note about DEN parameters.

TODO: include means to download pretrained generators.
```

## `dhs64_multiple_fsp.py`

Design enhancers for activity specific to two or more of the [DHS64-modeled biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) against all others. Uses Fast SeqProp. This script can be run with the following arguments:

```
Run Fast SeqProp to generate sequences with activity spcecific to multiple biosamples using DHS64.

This script can be run with the following arguments:

optional arguments:
  -h, --help            show this help message and exit
  --targets-idx TARGETS_IDX
                        Indices of biosamples to maximize among all DHS64-modeled biosamples, as a comma-separated list.
  --n-seqs N_SEQS       Number of sequences to generate.
  --seq-length SEQ_LENGTH
                        Length of sequences to generate.
  --output-dir OUTPUT_DIR
                        Directory to save output files.
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files. If None, a prefix based on target biosample indices and names will be used.
  --seed SEED           Random seed for sequence initialization. If None, a random seed will be used.
```

## `dhs64_single_tunable_fsp.py`

Design enhancers for activity specific to one of the [DHS64-modeled biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) against all others. Instead of maximizing target activity, this script optimizes sequences for a range of activity setpoints linearly interpolated between a specified minimum and maximum. A set of precomputed setpoing limits are included as defaults, covering what we found to be the Fast SeqProp-accessible range on each biosample.

This script can be run with the following arguments:

```
Run Fast SeqProp to generate sequences with biosample-specific activity using DHS64.

optional arguments:
  -h, --help            show this help message and exit
  --target-idx TARGET_IDX
                        Target biosample index within all DHS64-modeled biosamples.
  --setpoint-min SETPOINT_MIN
                        Minimum setpoint value for the target biosample. If not specified, a default value will be used.
  --setpoint-max SETPOINT_MAX
                        Maximum setpoint value for the target biosample. If not specified, a default value will be used.
  --n-seqs N_SEQS       Number of sequences to generate.
  --seq-length SEQ_LENGTH
                        Length of sequences to generate.
  --output-dir OUTPUT_DIR
                        Directory to save output files.
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files. If None, a prefix based on biosample index and name will be used.
  --seed SEED           Random seed for sequence initialization. If None, a random seed will be used.
```

This script generates the following additional output files:
- `{output-prefix}_setpoint_vs_preds_design_scatter.png`: Scatter plot showing the relationship between the intended setpoints and the predicted accessibilities of the designed sequences, as given by the design model.
- `{output-prefix}_setpoint_vs_preds_val_scatter.png`: Scatter plot showing the relationship between the intended setpoints and the predicted accessibilities of the designed sequences, as given by the validation model.

To reproduce the design of enhancers in the article, run the following:

```shell
python dhs64_single_tunable_fsp.py --target-idx {i} --n-seqs 120 --seq-length 145
```

for every value of `{i}` corresponding to the MPRA-assayed cell lines.

## `dhs733_single_fsp.py`

Design enhancers for activity specific to any cell type / biosample in the [DNase I Index](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2559-3/MediaObjects/41586_2020_2559_MOESM3_ESM.xlsx). Uses Fast SeqProp. 

We only design for non-redundant biosamples (i.e. with non-repeated biosample names) within the DNase I Index, resulting in 261 possible targets. The notebook `dhs733_create_output_transformation_matrix.ipynb` analyzes redundant biosamples and creates a transformation matrix `dhs733_nonredundant_transformation_matrix.npy` used to average redundant DHS733 outputs during design. It also creates the table `dhs733_nonredundant_biosample_metadata.tsv` which contains the final list of 261 targets. `--target-idx` corresponds to the zero-indexed position of a biosample within this table.

The script supports designing enhancers with different levels of stringency via the `--non-target-percentile` parameter, where a higher value may result in a more stringent enhancer at the cost of lower target activity.

This script can be run with the following arguments:

```
Run Fast SeqProp to generate sequences with biosample-specific activity using DHS733.

optional arguments:
  -h, --help            show this help message and exit
  --target-idx TARGET_IDX
                        Target biosample index within non-redundant DHS733-modeled biosamples. See "dhs733_nonredundant_biosample_metadata.tsv" for a list of possible target biosamples.
  --n-seqs N_SEQS       Number of sequences to generate.
  --seq-length SEQ_LENGTH
                        Length of sequences to generate.
  --non-target-percentile NON_TARGET_PERCENTILE
                        Percentile of non-target biosample predictions to explicitly minimize. Higher non-target percentile corresponds to more stringent designs.
  --output-dir OUTPUT_DIR
                        Directory to save output files.
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files. If None, a prefix based on biosample index and name will be used.
  --seed SEED           Random seed for sequence initialization. If None, a random seed will be used.
```

To reproduce the design of enhancers in the article, run the following:

```shell
python dhs733_single_fsp.py --target-idx {i} --n-seqs 50 --seq-length 145 --non-target-percentile {p}
```

for every value of `{i}` between 0 and 260, and `{p}` with values 85, 90, 95, or 98. 