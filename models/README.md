# Model training

## Downloading model weights

The following scripts can be run to download model weights:

- `download_dhs64_weights.sh`: DHS64 model weights.
- `download_dhs64_nofilt_weights.sh`: DHS64 model trained without filtering the training data (see below). Only used for model performance analysis and not recommended for any other workflow, particularly for enhancer design.
- `download_dhs64_mpra_weights.sh`: DHS64-MRPA model weights.
- `download_dhs733_weights.sh`: DHS733 model weights.

## Data processing

Processed DNase I Index data must be present for model training. See the documentation in the [`data`](/data/) folder.

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

Model training was originally performed in a `g5.2xlarge` AWS EC2 instance.

## DHS64-MPRA training

Coming soon!

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

Model training was originally performed in a `g5.xlarge` AWS EC2 instance.