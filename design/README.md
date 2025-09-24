# Model-based enhancer design scripts

The included scripts reproduce the enhancer design tasks in our article. These include:

- Design enhancers specific to **one biosample (i.e. tissue/cell type)** among [64 biosamples](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) captured by our initial model, DHS64: [`dhs64_single_fsp.py`](#dhs64_single_fsppy) and [`dhs64_single_den.py`](#dhs64_single_denpy).
- Design enhancers specific to **multiple biosamples** among those modeled by DHS64: [`dhs64_multiple_fsp.py`](#dhs64_multiple_fsppy).
- Design enhancers specific to one DHS64-modeled biosample with **tunable target activity**: [`dhs64_single_tunable.py`](#dhs64_single_tunable_fsppy).
- Design enhancers specific to **any of the hundreds of biosamples in the [DNase I Index](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2559-3/MediaObjects/41586_2020_2559_MOESM3_ESM.xlsx)**: [`dhs733_single_fsp.py`](#dhs733_single_fsppy).

Along with our accessibility models, these scripts use [Fast SeqProp](https://doi.org/10.1186/s12859-021-04437-5) and [Deep Exploration Networks (DENs)](https://doi.org/10.1016/j.cels.2020.05.007) to optimize sequences.

## Command-line interface
Scripts include a command-line interface which can generally be used as follows:

```shell
python {design_script}.py --target-idx {i} --n-seqs 100 --seq-length 145
```

Where `{i}` corresponds to the index of the biosample to target. See each individual subsection below for a precise description of each script's interface.

## Outputs

The output of each script is a directory with the following files:

- `{output-prefix}_4mer_distance.png`: Violin plot of the **normalized euclidean distance between 4mer counts** across generated sequences, as a measure of sequence diversity.
- `{output-prefix}_editdistance.png`: Violin plot of the **normalized edit distance** between generated sequences, as a measure of sequence diversity.
- `{output-prefix}_preds_design_boxplot.png`: Box plot of **predicted accessibilities** of generated sequences, as given by the **model used during sequence design**. This model is always an ensemble of two DHS64 or DHS733 models independently trained on distinct data splits.
- `{output-prefix}_preds_design.csv.gz`: Table with **predicted accessibilities** of generated sequences as given by the **design model**.
- `{output-prefix}_preds_val_boxplot.png`: Box plot of **predicted accessibilities** of generated sequences as given by a **validation model**. This model was independently trained on a distinct data split from the models used for design.
- `{output-prefix}_preds_val.csv.gz`: Table with **predicted accessibilities** of generated sequences as given by the **validation model**.
- `{output-prefix}_run_metadata.json`: Parameters used for sequence generation, including arguments provided to Fast SeqProp and DEN.
- `{output-prefix}_seq_bitmap.png`: Generated sequences represented as a bitmap, where rows correspond to individual sequences and pixels represent nucleotide identity (i.e. A, C, G, or T).
- `{output-prefix}_seqs.fasta`: **Generated sequences** in .fasta format.
- `{output-prefix}_seqs.png`: Logos of a few example generated sequences.
- `{output-prefix}_train_history.png`: Plots of the different **optimization losses** across iterations. Information about each individual loss can be found in the Fast SeqProp and DEN publications and their github repositories.

`output-prefix` can be specified via a command-line argument, otherwise it defaults to `{target_idx}_{target_biosample_name}`. Some scripts create additional files as indicated below.

## Code organization

These scripts can also be used as starting points for custom sequence design tasks. The code is organized as follows:

1. **Imports and constants**.
2. **Loss functions** defining sequence features to optimize. These include scores computed from model predictions (e.g. the difference between target and non-target accessibilities which we would like to maximize), as well as sequence features that we want to penalize such as long nucleotide repeats.
3. **Main sequence design function**. This function runs Fast SeqProp or trains DENs to generate sequences, makes and saves predictions using both the design and validation models, and creates various plots to analyze the generated sequences.
4. **Entry point** which parses command-line arguments and runs the main design function.

Note that most optimization-related arguments provided to Fast SeqProp and DEN are not currently exposed to the command-line API, and would need to be modified inside the code if needed.

# Description of individual scripts

## `dhs64_single_fsp.py`

Design enhancers with activity specific to **one out of [64 biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx)** captured by our initial model DHS64. Uses **Fast SeqProp**. 

This script can be run with the following arguments:

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

To reproduce the enhancer design task in the article, run the following:

```shell
python dhs64_single_fsp.py --target-idx {i} --n-seqs 250 --seq-length 145
```

for every value of `{i}` between 0 and 63 inclusive.

## `dhs64_single_den.py`

**Train a DEN** to generate enhancers with activity specific to **one out of [64 biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx)** captured by our initial model DHS64. 

The required file `dhs64_single_den_generator.py` defines the generator network architecture. Currently this architecture only allows for 145nt-long sequences, and should be modified if a different sequence length is needed.

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
- `{output-prefix}_generator.h5`: Trained generator.

To reproduce the enhancer design task in the article, run the following:

```shell
python dhs64_single_den.py --target-idx {i} --n-seqs 250
```

for every value of `{i}` between 0 and 63 inclusive.

This script also allows to use a previously trained DEN generator to generate new sequences. The script `dhs64_single_den_pretrained_145nt_download.py` downloads generators specific to each biosample trained as part of the article, along with parameters used for training.

## `dhs64_multiple_fsp.py`

Design enhancers with activity specific to **two or more out of the [64 biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx)** captured by our initial model DHS64. Uses Fast SeqProp. 

This script can be run with the following arguments:

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

To reproduce the double target enhancer design task in the article, run the following:

```shell
python dhs64_multiple_fsp.py --targets-idx {i,j} --n-seqs 120 --seq-length 145
```

For {i,j} including 49,61 (HepG2 + HeLaS3), 6,30 (NT2_D1 + SKNSH), 57,49 (K562 + HepG2), 45,30 (SJCRH30 + SKNSH), 49,31 (HepG2 + WERI_Rb1), 60,19 (MCF7 + 786_O), 57,17 (K562 + GM12878), and 30,17 (SKNSH + GM12878).

To reproduce the triple target enhancer design task in the article, run the following:

```shell
python dhs64_multiple_fsp.py --targets-idx {i,j,k} --n-seqs 120 --seq-length 145
```

For {i,j,k} including 30,17,19 (SKNSH + GM12878 + 786_O), 60,61,57 (MCF7 + HeLaS3 + K562), 6,61,31 (NT2_D1 + HeLaS3 + WERI_Rb1), 45,31,17 (SJCRH30 + WERI_Rb1 + GM12878), 61,49,60 (HeLaS3 + HepG2 + MCF7), 30,45,61 (SKNSH + SJCRH30 + HeLaS3), 49,17,6 (HepG2 + GM12878 + NT2_D1), and 49,57,30 (HepG2 + K562 + SKNSH).

## `dhs64_single_tunable_fsp.py`

Design enhancers for activity specific to **one out of [64 biosamples / cell types](./../data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx)** captured by our initial model DHS64. Instead of maximizing activity, this script optimizes enhancers for a **range of submaximal target activity setpoints** linearly interpolated between a specified minimum and maximum. A set of precomputed default setpoint limits are included, covering what we found to be the Fast SeqProp-enabled range on each biosample.

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
- `{output-prefix}_setpoint_vs_preds_design_scatter.png`: Scatter plot of **setpoints versus predicted accessibilities** of generated sequences, as given by the **design model**.
- `{output-prefix}_setpoint_vs_preds_val_scatter.png`: Scatter plot of **setpoints versus predicted accessibilities** of generated sequences, as given by the **validation model**.

To reproduce the tunable enhancer design task in the article, run the following:

```shell
python dhs64_single_tunable_fsp.py --target-idx {i} --n-seqs 120 --seq-length 145
```

for `{i}` including 6 (NT2_D1), 17 (GM12878), 19 (786_O), 30 (SKNSH), 31 (WERI_Rb1), 45 (SJCRH30), 49 (HepG2), 57 (K562), 60 (MCF7), and 61 (HeLaS3).

## `dhs733_single_fsp.py`

Design enhancers for activity specific to **any biosample in the [DNase I Index](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2559-3/MediaObjects/41586_2020_2559_MOESM3_ESM.xlsx)**, including **tissues**, **cell types**, and **cell states**. We only design for **non-redundant biosamples** (i.e. with unique biosample names) within the DNase I Index, resulting in **261 possible targets**. Uses Fast SeqProp.

The notebook `dhs733_create_output_transformation_matrix.ipynb` analyzes redundant biosamples and creates a transformation matrix `dhs733_nonredundant_transformation_matrix.npy` used to average redundant DHS733 outputs during design. This notebook also creates the table `dhs733_nonredundant_biosample_metadata.tsv` which contains the final list of 261 targets. `--target-idx` corresponds to the zero-indexed position of a biosample within this table.

The script supports designing enhancers with different levels of stringency via the `--non-target-percentile` parameter, where a higher value may result in a more stringent enhancer at the cost of lower target activity. The optimal value will depend on the cell type / biosample of interest.

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