# Programming human cell type-specific gene expression via an atlas of AI-designed enhancers

![plot](./readme_fig.png)

**Design synthetic enhancers with activity specific to the hundreds of human cell types, tissues, and differentiation states.** Our method uses AI predictors of chromatin accessibility trained on the [DNase I Index dataset](https://doi.org/10.1038/s41586-020-2559-3). The list of potential enhancer targets can be found [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2559-3/MediaObjects/41586_2020_2559_MOESM3_ESM.xlsx).

This repository is part of the following article: Castillo-Hair et al. *Programming human cell type-specific gene expression via an atlas of AI-designed enhancers* (link coming soon).

## Contents

This repository contains the following main components:

- **Sequence-to-function predictors**. The folder [`models`](./models) contains code to train predictors and to download pretrained weights. We use three predictor classes, each with three individual models independently trained on different chromosome-based data splits:
    
    - **DHS64**: *chromatin accessibility* predictor trained on a [subset of 64 samples](./data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) in the DNase I Index, selected to represent a wide variety of tissues and cell types.
    - **DHS733**: *chromatin accessibility* predictor trained on all 733 DNase I Index samples.
    - **DH64-MPRA**: *enhancer activity* predictor, developed by finetuning DHS64 models on MPRA data from 12 cell lines collected by us.
    
- **Enhancer design code**. The folder [`design`](./design/) contains code to generate synthetic enhancers using DHS64 or DHS733 as oracles. We use updated versions of **Fast SeqProp** ([paper](https://doi.org/10.1186/s12859-021-04437-5), [github](https://github.com/castillohair/corefsp/)) and **Deep Exploration Networks** ([paper](https://doi.org/10.1016/j.cels.2020.05.007), [github](https://github.com/castillohair/genesis/)) to optimize sequences. We include code to generate enhancers specific to one or multiple cell types, with maximal or tunable activites.

<!---
- **Analysis of experimental validation results**. We characterized the performance of ~9,000 enhancers, including synthetic ones and natural controls, via MPRAs in 10 target cell lines. The folder [`analysis`](./analysis/) contains code to analyze those results and generate figures in our publication.
-->

- **Atlas of human synthetic enhancers**. We include two repositories of synthetic enhancers designed with [DHS64]() (~32k, 500 per target) or [DHS733]() (~52.2k, 200 per non-redundant target), respectively, targeting each of their modeled cell types, along with pre-computed predictions.

Additional folders include:
- [`data`](./data): Data necessary for model training and analysis.
- [`src`](./src): python code used across the repository.

Individual folders contain their own README.md file with more specific information.

## Use cases

### Using pre-designed synthetic enhancers

Synthetic enhancers can be extracted from the included Atlas without running any code. In general, you will need to search for a cell type / tissue / cell state that most closely represents the desired target within the [DHS64](./data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx) and [DHS733](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-020-2559-3/MediaObjects/41586_2020_2559_MOESM3_ESM.xlsx) modeled samples, and then find relevant enhancers in the Atlas files.

Note that a subset of DHS64-designed enhancers has been experimentally validated in cell lines and mouse retina in our publication. We recommend preferentially using these if the cell type of interest can be adequately represented by any of these cell lines, and an experimentally characterized enhancer with the desired activity can be found.

### Designing new enhancers

Reasons to generate new enhancers beyond the Atlas may include: 1) different enhancer lengths, 2) submaximal target activity, 3) targeting multiple cell types.

To design new enhancers, install the [required packages](#requirements) and download the appropriate model weights into [`models`](./models/) using the included `download_model_weights.py` script. Then, scripts included in the [`design`](./design/) folder can be run with the appropriate settings or modified as needed.

In addition, we trained generative models called [Deep Exploration Networks](https://doi.org/10.1016/j.cels.2020.05.007) to target each of the DHS64-modeled samples. If the goal is to generate additional enhancers targeted to these samples, pre-trained DENs can be downloaded and used via included scripts.

See the [`design`](./design/) folder `README` for more information.

### Training models

#### Chromatin accessibility models

Model training can be performed via the `train.py` scripts included in the [`models/dhs64`](./models/dhs64/) and [`models/dhs733`](models/dhs733/) subfolders. These can be used to reproduce training of the models used in the article, as well as to train new models on additional data splits we did not originally use. See the [models](./models/) folder `README` file for more information.

Processed DNase I Index data used for training can be downloaded via the included `download_data.py` script in the [`data`](./data) folder. More information can be found [here](./data/README.md#data-for-training-accessibility-models).

#### Enhancer activity models

Finetuning of the DHS64-MPRA enhancer activity model can be reproduced via the `finetune.py` script in the [`models/dhs64_mpra`](./models/dhs64_mpra/) subfolder. This script can also be used to finetune on data splits not originally used in the article, and as a starting point to finetune on new MPRA measurements. See [DHS64-MPRA Finetuning](./models/README.md#dhs64-mpra-finetuning) for more information.

Processed MPRA measurements used for finetuning can be downloaded via the `download_data.py` script in the [`data`](./data/) folder. Necessary data split information will also be downloaded. More information can be found [here](./data/README.md#cell-line-and-mouse-retina-mpra-results).

<!---
### Reproduce publication analysis

Each analysis included in [`analysis`](./analysis/) will have its own workflow and requirements. See the folder's README.md file for more information.
-->

## Requirements

Code was written in Python 3 (tested in 3.7-3.10). Dependencies can be installed via `pip install -r requirements.txt`.

Notes about special requirements:

- **AI modeling packages**: All model-related code requires `tensorflow`,  Keras 2, and a compatible `tensorflow-probability`. The following version combinations have been tested:
    - `tensorflow` 2.7, `tensorflow-probability` 0.15.0
    - `tensorflow` 2.10, `tensorflow-probability` 0.15.0
    - `tensorflow` 2.14, `tensorflow-probability` 0.22.1

    These require `numpy` 1.x.
    
    If using other `tensorflow` versions, note that starting with v2.16, Keras 3 is included by default which may not work out of the box with this repository. Additionally, compatibility between `tensorflow` and `tensorflow-probability` should be verified at https://github.com/tensorflow/probability/releases.

- **Custom/modified packages for AI sequence design and interpretation**: The following should be installed from the linked repositories:
    - [`Fast SeqProp`](https://github.com/castillohair/corefsp): re-implemented compared to the original publication.
    - [`isolearn`](https://github.com/castillohair/isolearn): required by DEN, modified to work with tensorflow 2.
    - [`DEN`](https://github.com/castillohair/genesis): modified to work with tensorflow 2.
    - [`SHAP`](https://github.com/castillohair/shap): added modifications originally introduced by the Kundaje lab into a more recent SHAP version that works with tensorflow 2.

## Citation

If you use any of the contents of this repository, please cite the following:

Sebastian M. Castillo-Hair, Christopher H. Yin, Leah VandenBosch, Timothy J. Cherry, Wouter Meuleman, Georg Seelig. *Programming human cell type-specific gene expression via an atlas of AI-designed enhancers*.