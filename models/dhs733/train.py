import argparse
import os
import random
import sys

import numpy
import pandas
import tables

import tensorflow

BASE_DIR = '../../'
sys.path.append(BASE_DIR)
import src.definitions
import src.model
import src.sequence

DHS733_TRAIN_ONEHOT_SEQS_PATH = os.path.join(BASE_DIR, src.definitions.DHS733_TRAIN_ONEHOT_SEQS_PATH)
DHS733_TRAIN_LOGSIGNAL_PATH = os.path.join(BASE_DIR, src.definitions.DHS733_TRAIN_LOGSIGNAL_PATH)
DATA_SPLITS_DHS_IDX_PATH = os.path.join(BASE_DIR, src.definitions.DATA_SPLITS_DHS_IDX_PATH)

# Learning rate scheduler hyperparameters
lrs = [2e-4, 2e-5, 2e-6]
patience_r = [1, 3, 3]

# Loss weights
w_mse = 0.5
w_cosine = 0.5

class SeqGenerator(tensorflow.keras.utils.Sequence):
    """
    Generates data batches from a DHS dataframe.

    Parameters
    ----------
    seq_indices : list of int
        Indices of sequences to batch.
    randomize_padding : bool
        Whether to randomly choose left or right padding for each batch.
    randomize_rc : bool
        Whether to randomly reverse complement sequences for each batch.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the data at the end of each epoch.
        
    """
    def __init__(
            self,
            seq_indices,
            batch_size=128,
            randomize_padding=False,
            randomize_rc=False,
            shuffle=True,
        ):
        # Copy parameters
        self.seq_indices = seq_indices
        self.batch_size = batch_size
        self.randomize_padding = randomize_padding
        self.randomize_rc = randomize_rc
        self.shuffle = shuffle

        self.seq_oenehot_placeholder = numpy.zeros((batch_size, src.definitions.DHS733_INPUT_LENGTH, 4))
        self.signal_placeholder = numpy.zeros((batch_size, src.definitions.DHS733_N_BIOSAMPLES))
        
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch
        """
        return int(numpy.ceil(len(self.seq_indices) / self.batch_size))
    
    def load_onehot_seqs(self, seq_indices, padding='left', rc=False):
        """
        Load one-hot encoded sequences from file.

        Parameters
        ----------
        seq_indices : list of int
            Indices of sequences to load.
        padding : str
            'left' or 'right' padding.
        rc : bool
            Whether to return the reverse complement of the sequences.
        
        Returns
        -------
        numpy.ndarray
            One-hot encoded sequences of shape (len(seq_indices), max_seq_len, 4).
        
        """
        # if reverse complement, we need to get the opposite padding so that
        # the reverse complement ends up in the requested padding position
        if rc:
            if padding == 'left':
                padding = 'right'
            elif padding == 'right':
                padding = 'left'
        with tables.open_file(DHS733_TRAIN_ONEHOT_SEQS_PATH, mode='r') as f:
            if padding == 'left':
                onehot_seqs = f.root.seq_left[seq_indices, :, :]
            elif padding == 'right':
                onehot_seqs = f.root.seq_right[seq_indices, :, :]
        if rc:
            onehot_seqs = onehot_seqs[:, ::-1, ::-1]
        return onehot_seqs
    
    def load_signal(self, seq_indices):
        with tables.open_file(DHS733_TRAIN_LOGSIGNAL_PATH, mode='r') as f:
            signal = f.root.log_quantnorm_dhs_signal[seq_indices, :]
        return signal

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        batch_indices = self.shuffled_indices[index*self.batch_size:(index+1)*self.batch_size]

        # Sequences
        # Use pre-specified max sequence length
        # Randomly choose padding and reverse complement
        batch_seqs_onehot = self.load_onehot_seqs(
            batch_indices,
            padding=random.choice(['left', 'right']) if self.randomize_padding else 'left',
            rc=random.choice([True, False]) if self.randomize_rc else False,
        )

        # Signal values and labels
        batch_y = self.load_signal(batch_indices)

        return (
            batch_seqs_onehot,
            batch_y,
        )

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.shuffled_indices = self.seq_indices.copy()
        if self.shuffle == True:
            numpy.random.shuffle(self.shuffled_indices)

# Main training function
def train_model(
        data_split_idx=0,
        starting_model=None,
        output_name=None,
    ):
    
    # Make default output name if not provided
    # If continuing from a previous model and the default name is the same as the starting model,
    # add "_continued" to the output name.
    if output_name is None:
        output_name = f"dhs733_data_split_{data_split_idx}"
        if starting_model is not None and output_name==starting_model:
            output_name += "_continued"

    print(f"Training model {output_name}...")

    # Print message if a starting model is provided
    if starting_model is not None:
        print(f"Continuing training from model: {starting_model}")

    # Load sequence indices for current split
    print(f"Using data split {data_split_idx}.")
    with tables.open_file(DATA_SPLITS_DHS_IDX_PATH, mode='r') as f:
        train_split_group = f.get_node(f'/split_{data_split_idx}/train')
        seq_train_indices = train_split_group.indices[:]
        val_split_group = f.get_node(f'/split_{data_split_idx}/val')
        seq_val_indices = val_split_group.indices[:]
    seq_train_indices = numpy.where(seq_train_indices)[0]
    seq_val_indices = numpy.where(seq_val_indices)[0]

    print(f"Training set size: {len(seq_train_indices):,}")
    print(f"Validation set size: {len(seq_val_indices):,}")
    
    # Make model
    if starting_model is not None:
        model = src.model.load_model(f'{starting_model}.h5')
    else:
        model = src.model.make_resnet(
            src.definitions.DHS733_INPUT_LENGTH,
            groups=4,
            blocks_per_group=3,
            filters=480,
            kernel_size=13,
            dilation_rates=[1, 2, 4, 8],
            first_conv_activation='relu',
            n_outputs=src.definitions.DHS733_N_BIOSAMPLES,
            output_activation='linear',
        )
    
    # Compile the model and train
    epochs = 100
    batch_size = 256

    # Data generators
    generator_train = SeqGenerator(
        seq_train_indices,
        batch_size=batch_size,
        randomize_padding=True,
        randomize_rc=True,
    )
    generator_val = SeqGenerator(
        seq_val_indices,
        batch_size=batch_size,
    )

    # Add custom loss function and compile
    loss = src.model.MSE_Cosine_Loss(w_mse, w_cosine)
    adam = tensorflow.keras.optimizers.Adam(lrs[0])
    model.compile(loss=loss, optimizer=adam)

    # Callbacks: learning rate scheduler and checkpointing
    callbacks = [
        src.model.SequentialLearningScheduler(lrs, patience=patience_r),
        tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=output_name + '_checkpoint_{epoch:02d}.h5',
            save_weights_only=False,
            monitor='val_loss',
        ),
    ]

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
        '--data-split-idx',
        type=int,
        default=0,
        help='Data split index (0-9).',
    )
    parser.add_argument(
        '--starting-model',
        type=str,
        default=None,
        help='Name of starting model file to continue training from.',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Name of the output model. If not, a default name will be used.',
    )

    args = parser.parse_args()

    train_model(
        data_split_idx=args.data_split_idx,
        starting_model=args.starting_model,
        output_name=args.output_name,
    )