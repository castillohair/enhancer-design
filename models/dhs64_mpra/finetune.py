import argparse
import json
import os
import sys

import tensorflow

BASE_DIR = '../../'
sys.path.append(BASE_DIR)
import src.definitions
import src.model
import src.mpra
import src.sequence

MPRA_DATA_PATH = os.path.join(BASE_DIR, src.definitions.MPRA_DATA_PATH)
MPRA_DATA_SPLITS_PATH = os.path.join(BASE_DIR, src.definitions.MPRA_DATA_SPLITS_IDX_PATH)

# Hyperparameters
# Stage 1: train new output layer
stage1_lrs = [1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.125e-5, 1.5625e-5, 7.8125e-6, 3.90625e-6]
stage1_patience = 5

# Stage 2: finetune entire model
stage2_lrs = [1e-5, 5e-6, 2.5e-6, 1.25e-6, 6.25e-7, 3.125e-7, 1.5625e-7, 7.8125e-8, 3.90625e-8]
stage2_patience = 5

def finetune_model(
        data_split_idx=0,
        pretrained_model_path=None,
        output_name=None,
):
    # Make default pretrained model and output names
    if pretrained_model_path is None:
        pretrained_model_path = os.path.join(
            BASE_DIR,
            src.definitions.DHS64_MODEL_PATH[data_split_idx],
        )

    if output_name is None:
        output_name = f"dhs64_mpra_data_split_{data_split_idx}"
    else:
        if output_name.endswith('.h5'):
            output_name = output_name[:-3]

    print(f"Finetunuing DHS64-MPRA model on data split {data_split_idx}")
    print(f"Pretrained model: {pretrained_model_path}")
    print(f"Output model: {output_name}")

    ##########################
    # Load and preprocess data
    ##########################
    print('Loading and preprocessing data...')

    # Load MPRA data
    mpra_df = src.mpra.load_data(MPRA_DATA_PATH)

    # Load data split info
    with open(MPRA_DATA_SPLITS_PATH) as f:
        data_splits_info = json.load(f)
    data_split_info = data_splits_info[data_split_idx]

    # Extract training, validation, and test data
    mpra_train_df = mpra_df.loc[data_split_info['train']]
    mpra_valid_df = mpra_df.loc[data_split_info['val']]
    mpra_test_df = mpra_df.loc[data_split_info['test']]

    x_train = src.sequence.one_hot_encode(mpra_train_df['sequence'].values, max_seq_len=src.definitions.MODEL_INPUT_LENGTH)
    x_valid = src.sequence.one_hot_encode(mpra_valid_df['sequence'].values, max_seq_len=src.definitions.MODEL_INPUT_LENGTH)
    x_test = src.sequence.one_hot_encode(mpra_test_df['sequence'].values, max_seq_len=src.definitions.MODEL_INPUT_LENGTH)

    y_train = mpra_train_df[src.definitions.MPRA_CELL_LINES_LOG2FC_COLS].values
    y_valid = mpra_valid_df[src.definitions.MPRA_CELL_LINES_LOG2FC_COLS].values
    y_test = mpra_test_df[src.definitions.MPRA_CELL_LINES_LOG2FC_COLS].values

    #######################################################
    # Stage 1: train new output layer into pretrained model
    #######################################################
    print('Stage 1: training new output layer into pretrained model...')

    print('Loading pretrained model...')
    pretrained_model = src.model.load_model(pretrained_model_path)

    # Replace output with new output head for predicting log2FC measurements
    print('Adding new output head...')
    log2fc_output_head = tensorflow.keras.layers.Dense(
        len(src.definitions.MPRA_CELL_LINES_LOG2FC_COLS),
        activation='linear',
    )(pretrained_model.layers[-3].output)

    model = tensorflow.keras.models.Model(
        inputs=pretrained_model.input,
        outputs=log2fc_output_head,
    )

    # freeze all layers except for the last one
    for layer in model.layers[:-1]:
        layer.trainable = False
    
    # assert that only the last layer is trainable
    for layer in model.layers[-1:]:
        assert layer.trainable == True

    # Train model
    print('Training...')
    stage1_loss = src.model.MSE_Cosine_Loss(w_mse=0.5, w_cosine=0.5)
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(
            learning_rate=stage1_lrs[0], beta_1=0.9, beta_2=0.999),
        loss=stage1_loss,
    )
    stage1_lr_scheduler = src.model.SequentialLearningScheduler(
        stage1_lrs,
        patience=stage1_patience,
    )
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=250,
        validation_data=(x_valid, y_valid),
        callbacks=[stage1_lr_scheduler],
    )

    print('Stage 1 complete.')

    ############################
    # Stage 2: train whole model
    ############################
    print('Stage 2: finetuning entire model...')

    # Set trainable parameter but keep batch normalizaiton and droput layers non-trainable
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tensorflow.keras.layers.BatchNormalization) or isinstance(layer, tensorflow.keras.layers.Dropout):
            layer.trainable = False

    # Compile and train
    stage2_loss = src.model.MSE_Cosine_Loss(w_mse=0.5, w_cosine=0.5)
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(
            learning_rate=stage2_lrs[0], beta_1=0.9, beta_2=0.999),
        loss=stage2_loss,
    )
    stage2_lr_scheduler = src.model.SequentialLearningScheduler(
        stage2_lrs,
        patience=stage2_patience,
    )
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=250,
        validation_data=(x_valid, y_valid),
        callbacks=[stage2_lr_scheduler],
    )

    # Save model
    model.save(f'{output_name}.h5', save_traces=False, include_optimizer=False)
    print(f"Done with {output_name}.")

if __name__=='__main__':

    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-split-idx',
        type=int,
        default=0,
        help='Data split index (0-6).',
    )
    parser.add_argument(
        '--pretrained-model-path',
        type=str,
        default=None,
        help='Path to the pretrained model to start training from. ' \
        'If not specified, a DHS64 model corresponding to the specified ' \
        'data split will be used.',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Name of the output model. If not specified, ' \
        'a default name will be used.',
    )

    args = parser.parse_args()

    finetune_model(
        data_split_idx=args.data_split_idx,
        pretrained_model_path=args.pretrained_model_path,
        output_name=args.output_name,
    )