# Model training

## Downloading model weights

The following scripts can be run to download model weights:

- `dhs64/download_weights.sh`: DHS64 model weights.
- `dhs64/download_nofilt_weights.sh`: DHS64 model trained without filtering the training data (see below). Only used for model performance analysis and not recommended for any other workflow, particularly for enhancer design.
- `dhs64_mpra/download_weights.sh`: DHS64-MRPA model weights.
- `dhs733/download_weights.sh`: DHS733 model weights.

## Training data

For training accessibility models, processed DNase I Index data must be present in `data/dhs_index`. For finetuning DHS64-MPRA, MPRA results and precalculated data splits must be present in `data/mpra`. See the documentation in the [`data`](/data/) folder for more information.

## DHS64 training

The script `dhs64/train.py` can be used to reproduce model training or to train new models on additional data splits. The script accepts the following arguments:

```
usage: train.py [-h] [--data-split-idx DATA_SPLIT_IDX] [--numsamples-max NUMSAMPLES_MAX]
                [--starting-model STARTING_MODEL] [--output-name OUTPUT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --data-split-idx DATA_SPLIT_IDX
                        Data split index (0-9).
  --numsamples-max NUMSAMPLES_MAX
                        Maximum number of biosamples where DHSs used for training should be active.
  --starting-model STARTING_MODEL
                        Name of starting model file to continue training from.
  --output-name OUTPUT_NAME
                        Name of the output model. If not, a default name will be used.
```

Models were trained on data splits #0, #1, and #3, by filtering DHSs to be active in 10 or fewer DHS64-selected biosamples. To reproduce this, run the following:

```shell
python train.py --data-split-idx 0 --numsamples-max 10
python train.py --data-split-idx 1 --numsamples-max 10
python train.py --data-split-idx 3 --numsamples-max 10
```

For comparison purposes, we trained a model where DHSs were not filtered by the number of biosamples where they're active. To reproduce this, run the following:

```shell
python train.py --data-split-idx 3
```

Model training was originally performed in `g5.2xlarge` AWS EC2 instances containing an NVIDIA A10G Tensor Core GPU and 32 GiB RAM.

## DHS64-MPRA finetuning

The script `dhs64_mpra/finetune.py` can be used to reproduce model finetuning or to train new models on additional data splits. The script accepts the following arguments:

```
usage: finetune.py [-h] [--data-split-idx DATA_SPLIT_IDX] [--pretrained-model-path PRETRAINED_MODEL_PATH]
                   [--output-name OUTPUT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --data-split-idx DATA_SPLIT_IDX
                        Data split index (0-6).
  --pretrained-model-path PRETRAINED_MODEL_PATH
                        Path to the pretrained model to start training from. If not specified, a DHS64 model
                        corresponding to the specified data split will be used.
  --output-name OUTPUT_NAME
                        Name of the output model. If not specified, a default name will be used.
```

In the article, finetuning on MPRA data splits #0, #1, and #3 was performed starting from DHS64 models trained on DHS data splits #0, #1, and #3 respectively. To reproduce this, run the following:

```shell
python finetune.py --data-split-idx 0
python finetune.py --data-split-idx 1
python finetune.py --data-split-idx 3
```

## DHS733 training

The script `dhs733/train.py` can be used to reproduce model training or to train new models on additional data splits. The script accepts the following arguments:

```
usage: train.py [-h] [--data-split-idx DATA_SPLIT_IDX] [--starting-model STARTING_MODEL] [--output-name OUTPUT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --data-split-idx DATA_SPLIT_IDX
                        Data split index (0-9).
  --starting-model STARTING_MODEL
                        Name of starting model file to continue training from.
  --output-name OUTPUT_NAME
                        Name of the output model. If not, a default name will be used.
```

Models were trained on data splits #0, #1, and #3. To reproduce this, run the following:

```shell
python train.py --data-split-idx 0
python train.py --data-split-idx 1
python train.py --data-split-idx 3
```

Model training was originally performed in `g5.xlarge` AWS EC2 instances containing an NVIDIA A10G Tensor Core GPU and 16 GiB RAM.
