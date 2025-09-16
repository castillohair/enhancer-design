import argparse
import json
import os
import random
import sys

import numpy
import pandas

import tensorflow

import Bio
import Bio.Seq

BASE_DIR = '../../'
sys.path.append(BASE_DIR)
import utils.definitions
import utils.resnet
import utils.sequence

DHS64_TRAIN_DATA_PATH = os.path.join(BASE_DIR, utils.definitions.DHS64_TRAIN_DATA_PATH)
DATA_SPLITS_CHRS_PATH = os.path.join(BASE_DIR, utils.definitions.DATA_SPLITS_CHRS_PATH)

class SeqGenerator(tensorflow.keras.utils.Sequence):
    """
    Generates data batches from a DHS dataframe.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DHS data to batch. Must include columns:
        - ('metadata', 'raw_sequence'): raw DNA sequences (str)
        - 'continuous': continuous log-transformed accessibility values
        - 'binary': binary peak calls
    batch_size : int
        Number of samples per batch.
    max_seq_len : int
        Maximum sequence length for one-hot encoding.
    shuffle : bool
        Whether to shuffle the data at the end of each epoch.
        
    """
    def __init__(
            self,
            data_df,
            batch_size=128,
            max_seq_len=500,
            shuffle=True,
        ):
        # Copy parameters
        self.data_df = data_df
        self.batch_size = batch_size
        self.max_seq_len=max_seq_len
        self.shuffle = shuffle
        
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch
        """
        return int(numpy.ceil(len(self.data_df) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Extract batch data
        batch_table = self.data_df.iloc[batch_indexes]

        # Sequences
        # Use pre-specified max sequence length
        # Padding will be chosen randomly on each batch
        batch_seqs = batch_table[('metadata', 'raw_sequence')].values.reshape(-1)
        batch_seqs_onehot = utils.sequence.one_hot_encode(
            batch_seqs, 
            max_seq_len=self.max_seq_len, 
            mask_val=0,
            padding=random.choice(['left', 'right']),
        )

        # Signal values and labels
        batch_y_reg = batch_table['continuous'].values
        batch_y_class = batch_table['binary'].values

        return (
            batch_seqs_onehot,
            [batch_y_reg, batch_y_class],
        )

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = numpy.arange(len(self.data_df))
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)

# Callback to save intermediate models after each epoch
class ModelCheckpoint(tensorflow.keras.callbacks.Callback):
    def __init__(self, model_to_save, filepath):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.model_to_save = model_to_save
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        # logs['val_loss']
        filepath = self.filepath.format(epoch=epoch)
        self.model_to_save.save(filepath)

# Main training function
def train_model(
        chr_split_idx=0,
        numsamples_max=None,
        starting_model=None,
        output_name=None,
    ):
    
    # Make default output name if not provided
    # If continuing from a previous model and the default name is the same as the starting model,
    # add "_continued" to the output name.
    if output_name is None:
        output_name = f"dhs64_chr_split_{chr_split_idx}"
        if starting_model is not None and output_name==starting_model:
            output_name += "_continued"

    print(f"Training model {output_name}...")

    # Print message if a starting model is provided
    if starting_model is not None:
        print(f"Continuing training from model: {starting_model}")

    # Load training data
    print("Loading training data...")
    dhs_df = pandas.read_csv(
        DHS64_TRAIN_DATA_PATH,
        header=[0, 1],
        index_col=0,
        dtype = {'identifier': str},
    )
    dhs_df.rename(columns={'nan': numpy.nan}, inplace=True)
    print(f"{len(dhs_df):,} DHSs loaded.")

    # Extract biosamples
    biosamples = dhs_df['continuous'].columns.tolist()
    print("Biosamples:")
    print(biosamples)

    # Load chromsome split
    print(f"Separating data by chromosome split. Using split {chr_split_idx}.")

    with open(DATA_SPLITS_CHRS_PATH, 'r') as f:
        chr_splits_info = json.load(f)
    chr_split_info = chr_splits_info[chr_split_idx]
    
    print(f"Chromsomes for training: {chr_split_info['train']}")
    print(f"Chromsomes for validation: {chr_split_info['val']}")
    print(f"Chromsomes for testing: {chr_split_info['test']}")

    # Separate training and validation datasets
    dhs_df_train = dhs_df[dhs_df[('metadata', 'seqname')].isin(chr_split_info['train'])]
    dhs_df_val = dhs_df[dhs_df[('metadata', 'seqname')].isin(chr_split_info['val'])]

    print(f"{len(dhs_df_train):,} DHS sites loaded for training.")
    print(f"{len(dhs_df_val):,} DHS sites loaded for validation.")

    # Filtering by number of samples if specified
    if numsamples_max is not None:
        print(f"Filtering training data to keep only DHSs active in at most {numsamples_max} biosamples...")
        dhs_df_train = dhs_df_train[
            (dhs_df_train[('metadata', 'numsamples_selected')] > 0) &
            (dhs_df_train[('metadata', 'numsamples_selected')] <= numsamples_max)
        ]
        print(f"{len(dhs_df_train):,} DHSs remain for training after filtering.")
        dhs_df_val = dhs_df_val[
            (dhs_df_val[('metadata', 'numsamples_selected')] > 0) &
            (dhs_df_val[('metadata', 'numsamples_selected')] <= numsamples_max)
        ]
        print(f"{len(dhs_df_val):,} DHSs remain for validation after filtering.")

    # Add reverse complement to training data
    # Get reverse complement of sequences, then construct dataframe
    # with rev. comp. and measurements of original sequences.
    # Finally, concatenate the original and the new rev. comp. dataframe
    rev_comp_seqs = dhs_df_train[('metadata', 'raw_sequence')].map(Bio.Seq.reverse_complement)
    dhs_df_train_rc = dhs_df_train.copy()
    dhs_df_train_rc[('metadata', 'raw_sequence')] = rev_comp_seqs
    dhs_df_train = pandas.concat([dhs_df_train, dhs_df_train_rc], axis=0)
    
    # Make model
    if starting_model is not None:
        model = utils.resnet.load_model(f'{starting_model}.h5')
    else:
        model = utils.resnet.make_model(
            utils.definitions.MODEL_MAX_SEQ_LEN,
            groups=4,
            blocks_per_group=3,
            filters=256,
            kernel_size=13,
            dilation_rates=[1, 2, 4, 8],
            first_conv_activation='relu',
            n_outputs=len(biosamples),
            output_activation=['linear', 'sigmoid'],
        )
    
    # Compile the model and train
    epochs = 100
    batch_size = 256

    # Data generators
    generator_train = SeqGenerator(
        dhs_df_train,
        batch_size=batch_size,
        max_seq_len=utils.definitions.MODEL_MAX_SEQ_LEN,
    )
    generator_val = SeqGenerator(
        dhs_df_val,
        batch_size=batch_size,
        max_seq_len=utils.definitions.MODEL_MAX_SEQ_LEN,
    )

    # Use Adam optimizer
    learning_rate = 2e-4
    adam = tensorflow.keras.optimizers.Adam(learning_rate)

    # Callback: stop training after no improvement in validation
    callbacks = [
        tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            model_to_save=model,
            filepath=output_name + '_checkpoint_{epoch:03d}.h5',
        )
    ]
    
    # Weights for losses
    regression_loss_weight = 1 / len(biosamples)
    classification_loss_weight = 1 / len(biosamples)

    # Compile
    model.compile(
        optimizer=adam,
        loss=['mse', 'binary_crossentropy'],
        loss_weights=[regression_loss_weight, classification_loss_weight],
    )

    # Fit
    model.fit(
        x=generator_train,
        validation_data=generator_val,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    model.save(f'{output_name}.h5', save_traces=False, include_optimizer=False)

    print(f"Done with {output_name}.")

if __name__=='__main__':

    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chr-split-idx', type=int, default=0, help='Chromosome split index (0-9)',
    )
    parser.add_argument(
        '--numsamples-max', type=int, default=None, help='Maximum number of biosamples where DHSs used for training should be active',
    )
    parser.add_argument(
        '--starting-model', type=str, default=None, help='Path to starting model (if any)',
    )
    parser.add_argument(
        '--output-name', type=str, default=None, help='Name of the output model. If not, a default name will be used.',
    )

    args = parser.parse_args()

    train_model(
        chr_split_idx=args.chr_split_idx,
        numsamples_max=args.numsamples_max,
        starting_model=args.starting_model,
        output_name=args.output_name,
    )