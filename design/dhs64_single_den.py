"""
Generate DNA sequences with chromatin accessibility specific to a target biosample, using
the DHS64 model. This script uses Deep Exploration Networks (DENs), which are generative
models that need to be trained for each biosample target. Thus, a new generator model is
trained from scratch unless a pre-trained generator is provided.

Script sections:
1. Imports and constants: Import necessary libraries and define paths to model files and
   metadata. Note that the dhs64_single_den_generator module is required for building and
   loading the DEN generator model. Currently, the generator dimensions are hardcoded to
   support only 145bp sequences. Other sequence lengths would require modifying the
   generator architecture in dhs64_single_den_generator.py.
2. Loss functions: Specify sequence features to optimize. We define a target loss function
   that maximizes target biosample accessibility and minimizes average accessibilities
   across non-target biosamples. Additional loss functions required for DEN training
   include an entropy loss to encourage low-entropy (i.e. close to one-hot) PWMs and a
   similarity loss to encourage diversity between sequences generated from two different
   latent vectors.
3. Core DEN training function: Builds and trains a DEN generator model using the specified
   loss functions and a pessimistic ensemble of design models. Saves the trained generator,
   training history, and plots of the training history.
4. Main sequence design function: Trains a DEN generator if a pre-trained generator is not
   provided, generates sequences using the trained or provided generator, makes predictions
   using both the design and validation models, and creates various plots to analyze the
   generated sequences.
5. Entry point: Parses command-line arguments and runs the main design function.

"""

import argparse
import datetime
import functools
import json
import os
import sys

import numpy
import matplotlib
from matplotlib import pyplot
import pandas
import seaborn

matplotlib.rcParams['savefig.dpi'] = 120
matplotlib.rcParams['savefig.bbox'] = 'tight'

import tensorflow

import genesis
import genesis.generator
import genesis.predictor
import genesis.optimizer

BASE_DIR = '../'
sys.path.append(BASE_DIR)
import src.definitions
import src.model
import src.sequence
import src.plot
import src.utils

import dhs64_single_den_generator

BIOSAMPLE_META_PATH = os.path.join(BASE_DIR, src.definitions.DHS64_BIOSAMPLE_META_PATH)
DESIGN_MODEL_PATHS = [os.path.join(BASE_DIR, src.definitions.DHS64_MODEL_PATH[i]) for i in [1, 3]]
VAL_MODEL_PATH = os.path.join(BASE_DIR, src.definitions.DHS64_MODEL_PATH[0])

##################
# Loss functions #
##################

def get_fitness_loss(
        target_idx=0,
        fitness_min_weight=1.0,
        fitness_max_weight=1.0,
    ):
    """
    Get fitness loss function for DEN training.

    The returned function maximizes the predicted difference between target biosample
    and the average of non-target biosamples.

    Parameters
    ----------
    target_idx : int
        The index of the target biosample within the model outputs.
    fitness_min_weight : float
        Weight for minimizing non-target biosample predictions.
    fitness_max_weight : float
        Weight for maximizing target biosample prediction.

    Returns
    -------
    fitness_loss_func : function
        A function that takes loss tensors as input and returns the fitness loss.

    """
    # fitness loss function
    def fitness_loss_func(loss_tensors) :
        # Unpack loss tensors
        _, _, _, sequence_class, \
        pwm_logits_1, pwm_logits_2, pwm_1, pwm_2, \
        sampled_pwm_1, sampled_pwm_2, mask, sampled_mask, \
        pred = loss_tensors
        
        # Dimensions of pred: (batch_size, n_samples, n_model_outputs)
        # Create masks based on the specified index
        n_model_outputs = pred.shape[-1]
        target_mask = numpy.zeros((1, 1, n_model_outputs))
        target_mask[:, :, target_idx] = 1
        target_mask = tensorflow.cast(target_mask, tensorflow.float32)

        fitness_loss = \
            fitness_min_weight*tensorflow.keras.backend.mean((1 - target_mask)*pred, axis=-1) \
            - fitness_max_weight*tensorflow.keras.backend.sum(target_mask*pred, axis=-1)
        fitness_loss = tensorflow.keras.backend.mean(fitness_loss, axis=-1)

        return fitness_loss
    
    return fitness_loss_func

def get_sequence_loss():
    """
    Get sequence loss function for DEN training.

    The returned function is a placeholder that returns zero loss.

    Returns
    -------
    seq_loss_func : function
        A function that takes loss tensors as input and returns zero loss.

    """
    # sequence loss function
    def seq_loss_func(loss_tensors) :
        # Unpack inputs
        _, _, _, sequence_class, \
        pwm_logits_1, pwm_logits_2, pwm_1, pwm_2, \
        sampled_pwm_1, sampled_pwm_2, mask, sampled_mask, \
        pred = loss_tensors
        
        # constant zero tensor of shape (batch_size,)
        # get batch size from pwm_1
        seq_loss = 0*tensorflow.keras.backend.sum(pwm_1, axis=[1,2,3])
        
        return seq_loss
    
    return seq_loss_func

def get_entropy_loss(
        entropy_target_bits=1.8,
    ):
    """
    Get entropy loss function for DEN training.

    The returned function encourages the generated PWMs to have a low entropy
    per position.

    Parameters
    ----------
    entropy_target_bits : float
        Margin in bits for the entropy loss.

    Returns
    -------
    entropy_loss_func : function
        A function that takes loss tensors as input and returns the entropy loss.

    """
    # Entropy loss function
    entropy_loss_func = genesis.optimizer.get_margin_entropy_ame(
        pwm_start=0,
        pwm_end=None,
        min_bits=entropy_target_bits,
    )

    def loss_func(loss_tensors) :
        # Unpack inputs
        _, _, _, sequence_class, \
        pwm_logits_1, pwm_logits_2, pwm_1, pwm_2, \
        sampled_pwm_1, sampled_pwm_2, mask, sampled_mask, \
        pred = loss_tensors
        
        # Entropy loss
        entropy_loss = entropy_loss_func(pwm_1)
        
        return entropy_loss
    
    return loss_func

def get_similarity_loss(
        similarity_seq_margin=0.5,
    ):
    """
    Get similarity loss function for DEN training.

    The returned function uses a cosine similarity function between sampled sequences
    from the two generated PWMs as a penalty to encourage diversity. It also uses
    the similarity between 1nt-shifted versions of the sequences.

    Parameters
    ----------
    similarity_seq_margin : float
        Margin for the similarity loss.

    Returns
    -------
    similarity_loss_func : function
        A function that takes loss tensors as input and returns the similarity loss.

    """
    # Similarity loss function
    seq_similarity_func = genesis.optimizer.get_pwm_margin_sample_entropy(
        pwm_start=0,
        pwm_end=None,
        margin=similarity_seq_margin,
        shift_1_nt=True,
    )

    def loss_func(loss_tensors) :
        # Unpack inputs
        _, _, _, sequence_class, \
        pwm_logits_1, pwm_logits_2, pwm_1, pwm_2, \
        sampled_pwm_1, sampled_pwm_2, mask, sampled_mask, \
        pred = loss_tensors
        
        # Similarity loss
        similarity_loss = tensorflow.keras.backend.mean(
            seq_similarity_func(sampled_pwm_1, sampled_pwm_2),
            axis=1,
        )
        
        return similarity_loss
    
    return loss_func

############################
# Core DEN training function
############################

def train_den(
        seq_length,
        target_idx,
        models_design_filepath,
        fitness_loss_weight=1.0,
        seq_loss_weight=1.0,
        entropy_loss_weight=1.0,
        similarity_loss_weight=1.0,
        fitness_loss_params={},
        seq_loss_params={},
        entropy_loss_params={},
        similarity_loss_params={},
        learning_rate=0.001,
        batch_size=32,
        n_samples=1,
        n_epochs=10,
        steps_per_epoch=100,
        random_seed=None,
        enable_op_determinism=False,
        output_dir='.',
        output_prefix='den',
    ):
    """
    Train a DEN generator model to design sequences with biosample-specific activity.

    Outputs of this function include the trained generator model, a CSV file with
    training history, and a PNG file with plots of the training history.

    Parameters
    ----------
    seq_length : int
        Length of sequences to generate.
    target_idx : int
        Index of the target biosample within the model outputs.
    models_design_filepath : list of str
        Filepaths to predictors used during design.
    fitness_loss_weight, seq_loss_weight, entropy_loss_weight, similarity_loss_weight : float
        Weights for each loss function term.
    fitness_loss_params, seq_loss_params, entropy_loss_params, similarity_loss_params : dict
        Parameters for the construction of each loss function term.
    learning_rate : float
        Learning rate for generator training.
    batch_size : int
        Number of PWMs to generate per gradient step.
    n_samples : int
        Number of one-hot sequences to sample from the PWM at each gradient step.
    n_epochs : int
        Number of epochs to train the generator.
    steps_per_epoch : int
        Number of steps (gradient updates) per epoch.
    random_seed : int or None
        Random seed for training. If None, no seed is set.
    enable_op_determinism : bool
        Whether to enable TensorFlow operation determinism. If enabled, model training is
        reproducible from the seed number, but training is slower.
    output_dir : str
        Directory to save output files.
    output_prefix : str
        Prefix for output files.
    
    """

    # Set seeds before model creation
    if random_seed is not None:
        numpy.random.seed(random_seed)
        tensorflow.keras.utils.set_random_seed(random_seed)
    if enable_op_determinism:
        tensorflow.config.experimental.enable_op_determinism()
    
    # Load design model ensemble
    # Load individual design models
    models_design_list = [src.model.load_model(filepath) for filepath in models_design_filepath]
    # Select first output head: continous accessibility prediction
    models_design_list = [src.model.select_output_head(m, 0) for m in models_design_list]
    n_model_outputs = models_design_list[0].output.shape[-1]
    # Pessimistic ensemble: minimum across target biosample, maximum across non-target biosamples
    model_design = src.model.make_model_ensemble(
        models_design_list,
        min_output_idx=[target_idx],
        max_output_idx=[i for i in range(n_model_outputs) if i != target_idx],
        padded_input_length=seq_length,
    )
    # Design model should not be trainable
    model_design.trainable = False
    
    # Build DEN Generator Network
    _, generator = genesis.generator.build_generator(
        batch_size=batch_size,
        seq_length=seq_length,
        load_generator_function=dhs64_single_den_generator.make_generator,
        n_classes=1,
        n_samples=n_samples,
        sequence_templates=['N'*seq_length],
        batch_normalize_pwm=False,
    )
    
    # Build DEN "Predictor" Network, consisting of generator + design model
    # The "build_predictor" function needs a "load_predictor_function" that takes a sequence input
    # and a sequence class input, and returns additional predictor inputs (not used here), model
    # outputs, and a weight initialization function (not used here).
    # We will define a "load_predictor_function" with the right signature using the pre-loaded model.
    def load_predictor_function(sequence_input, sequence_class):
        model_input = tensorflow.keras.layers.Reshape((-1, 4))(sequence_input)
        model_output = model_design(model_input)
        return [], [model_output], lambda x: None
    # Build predictor
    _, predictor = genesis.predictor.build_predictor(
        generator_model=generator,
        load_predictor_function=load_predictor_function,
        batch_size=batch_size,
        n_samples=n_samples,
        eval_mode='sample',
    )
    
    # Get loss functions
    fitness_loss_func = get_fitness_loss(**fitness_loss_params)
    seq_loss_func = get_sequence_loss(**seq_loss_params)
    entropy_loss_func = get_entropy_loss(**entropy_loss_params)
    similarity_loss_func = get_similarity_loss(**similarity_loss_params)

    # Build loss model
    loss_model = tensorflow.keras.models.Model(
        predictor.inputs,
        [
            tensorflow.keras.layers.Lambda(lambda out: fitness_loss_func(out), name='fitness')(predictor.inputs + predictor.outputs),
            tensorflow.keras.layers.Lambda(lambda out: seq_loss_func(out), name='sequence')(predictor.inputs + predictor.outputs),
            tensorflow.keras.layers.Lambda(lambda out: entropy_loss_func(out), name='entropy')(predictor.inputs + predictor.outputs),
            tensorflow.keras.layers.Lambda(lambda out: similarity_loss_func(out), name='similarity')(predictor.inputs + predictor.outputs),
        ],
    )

    # Compile loss model
    def get_weighted_loss(loss_coeff=1.):
        def _min_pred(y_true, y_pred):
            return loss_coeff * y_pred

        return _min_pred
    
    loss_model.compile(
        loss={
            'fitness': get_weighted_loss(fitness_loss_weight),
            'sequence': get_weighted_loss(seq_loss_weight),
            'entropy': get_weighted_loss(entropy_loss_weight),
            'similarity': get_weighted_loss(similarity_loss_weight),
        },
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
    )

    # Run training
    fake_inputs = [
        numpy.ones((batch_size*steps_per_epoch, 1)),   # sequence_class_input
        numpy.ones((batch_size*steps_per_epoch, 100)), # latent_input_1
        numpy.ones((batch_size*steps_per_epoch, 100)), # latent_input_2
    ]
    fake_outputs = [
        numpy.ones((batch_size*steps_per_epoch, n_model_outputs)),
        numpy.ones((batch_size*steps_per_epoch, n_model_outputs)),
        numpy.ones((batch_size*steps_per_epoch, n_model_outputs)),
        numpy.ones((batch_size*steps_per_epoch, n_model_outputs)),
    ]
    train_history = loss_model.fit(
        x=fake_inputs,
        y=fake_outputs,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
    )

    # Replace input layers on generator to be used for generation
    generator.get_layer('lambda_rand_sequence_class').function = lambda inp: inp
    generator.get_layer('lambda_rand_input_1').function = lambda inp: inp
    generator.get_layer('lambda_rand_input_2').function = lambda inp: inp

    # Save models
    generator_path = os.path.join(output_dir, f'{output_prefix}_generator.h5')
    generator.save(generator_path)

    # Save training history
    train_history_df = pandas.DataFrame(train_history.history)
    # Normalize out weights
    if fitness_loss_weight > 0:
        train_history_df['fitness_loss'] /= fitness_loss_weight
    if seq_loss_weight > 0:
        train_history_df['sequence_loss'] /= seq_loss_weight
    if entropy_loss_weight > 0:
        train_history_df['entropy_loss'] /= entropy_loss_weight
    if similarity_loss_weight > 0:
        train_history_df['similarity_loss'] /= similarity_loss_weight
    train_history_df.index.name = 'epoch'
    # Save dataframe
    train_history_filepath = os.path.join(output_dir, f'{output_prefix}_training_history.csv')
    train_history_df.to_csv(train_history_filepath)

    # Plot training history
    fig, axes = pyplot.subplots(1, train_history_df.shape[1], figsize=(3.5*train_history_df.shape[1], 2.5))
    fig.subplots_adjust(wspace=0.4)
    for i, col in enumerate(train_history_df.columns):
        ax = axes[i]
        seaborn.lineplot(
            data=train_history_df,
            x=train_history_df.index,
            y=col,
            ax=ax,
        )
        ax.set_xlabel('Epoch')
        ax.set_title(col)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_training_history.png"))

#################################
# Main sequence design function #
#################################

def run(
        target_idx,
        n_seqs,
        seq_length=145,
        output_dir='.',
        output_prefix=None,
        generator_path=None,
        random_seed_train=None,
        random_seed_gen=None,
    ):
    """
    Run sequence design for a specified biosample using DEN.

    This function trains a DEN generator if a pre-trained generator is not provided,
    generates sequences using the trained or provided generator, makes predictions
    using both the design and validation models, and creates various plots to analyze
    the generated sequences.

    Parameters
    ----------
    target_idx : int
        Index of the target biosample within all DHS64-modeled biosamples.
    n_seqs : int
        Number of sequences to generate.
    seq_length : int
        Length of sequences to generate.
    output_dir : str
        Directory to save output files.
    output_prefix : str or None
        Prefix for output files. If None, a prefix based on biosample index and name will be used.
    generator_path : str or None
        Path to a pre-trained generator model. If None, a new generator will be trained.
    random_seed_train : int or None
        Random seed for generator training. If None, no seed is set.
    random_seed_gen : int or None
        Random seed for sequence generation. If None, no seed is set.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load metadata of modeled biosamples
    biosample_metadata_df = pandas.read_excel(BIOSAMPLE_META_PATH)
    biosamples = biosample_metadata_df['Biosample name'].tolist()

    # Extract and sanitize biosample name
    target = biosamples[target_idx]
    target_sanitized = src.utils.sanitize_str(target)

    # Prefix for output files
    if output_prefix is None:
        output_prefix = f"{target_idx}_{target_sanitized}"

    print(f"\nStarting sequence design for biosample {target} ({target_idx} / {len(biosamples)})...")

    # Generator training
    # ==================
    if generator_path is None:
        print(f"\nTraining DEN generator...")

        train_parameters = {
            # Target biosample index
            'target_idx': target_idx,
            # Length of sequences to generate
            'seq_length': seq_length,
            # Path of saved predictors
            'models_design_filepath': DESIGN_MODEL_PATHS,
            
            # Parameters for DEN generator training
            # Learning rate
            'learning_rate': 0.0002,
            # Number of PWMs to generate per grad step
            'batch_size': 32,
            # Number of one-hot sequences to sample from the PWM at each grad step
            'n_samples': 1,
            # Number of epochs
            'n_epochs': 50,
            # Number of steps (grad updates) per epoch
            'steps_per_epoch': 500,

            # Where to save output files
            'output_dir': output_dir,
            'output_prefix': output_prefix,
            
            # Weights for each loss function term
            'fitness_loss_weight': 1.0,
            'seq_loss_weight': 0.0,
            'entropy_loss_weight': 1.0,
            'similarity_loss_weight': 5.0,

            # Parameters for each loss function term
            'fitness_loss_params': {
                'target_idx': target_idx,
                'fitness_min_weight': 0.3,
                'fitness_max_weight': 0.5,
            },
            'seq_loss_params': {
            },
            'entropy_loss_params': {
                'entropy_target_bits': 1.8,
            },
            'similarity_loss_params': {
                'similarity_seq_margin': 0.3,
            },

            # Random seed for training
            'random_seed': random_seed_train,
        }
        with open(os.path.join(output_dir, f'{output_prefix}_train_metadata.json'), 'w') as file:
            file.write(json.dumps(train_parameters, indent=4))
            
        train_den(**train_parameters)
        print(f"Generator training complete for biosample {target} ({target_idx} / {len(biosamples)}).")

    else:
        print(f"\nSkipping generator training and loading from {generator_path}...")

    # Sequence generation and analysis
    # ================================
    print(f"\nGenerating sequences...")

    # Load trained generator
    if generator_path is None:
        generator_path = os.path.join(output_dir, f'{output_prefix}_generator.h5')
    generator_model = dhs64_single_den_generator.load_generator(generator_path)

    # Generate sequences and make plots for each cell type target
    numpy.random.seed(random_seed_gen)
    batch_size = generator_model.input[0].shape[0]
    n_seqs_ceil = int(numpy.ceil(n_seqs/batch_size)*batch_size)
    sequence_class = numpy.array([0] * n_seqs_ceil).reshape(-1, 1) 
    latent_1 = numpy.random.uniform(-1, 1, (n_seqs_ceil, 100))
    latent_2 = numpy.random.uniform(-1, 1, (n_seqs_ceil, 100))

    # Generate and save sequences
    pred_outputs = generator_model.predict(
        [sequence_class, latent_1, latent_2],
        batch_size=batch_size,
    )
    _, _, _, _, _, sampled_pwm, _, _, _ = pred_outputs
    generated_onehot = sampled_pwm[:n_seqs, 0, :, :, 0]
    generated_seqs = src.sequence.one_hot_decode(generated_onehot)
    generated_seq_ids = [f'{output_prefix}_seq_{i}' for i in range(len(generated_seqs))]
    generated_df = pandas.DataFrame(
        {'seq_id': generated_seq_ids, 'seq': generated_seqs}
    )
    src.sequence.save_seqs_to_fasta(
        generated_df,
        os.path.join(output_dir, f"{output_prefix}_seqs.fasta"),
        id_col='seq_id',
        seq_col='seq',
    )

    print(f"\nGenerating predictions...")

    # Generate and save predictions from design model
    # Load individual design models and construct ensemble
    models_design_list = [src.model.load_model(filepath) for filepath in DESIGN_MODEL_PATHS]
    # Select first output head: continous accessibility prediction
    models_design_list = [src.model.select_output_head(m, 0) for m in models_design_list]
    # Pessimistic ensemble: minimum across target biosample, maximum across non-target biosamples
    min_output_idx = [target_idx]
    max_output_idx = [i for i in range(len(biosamples)) if i != target_idx]
    model_design = src.model.make_model_ensemble(
        models_design_list,
        min_output_idx=min_output_idx,
        max_output_idx=max_output_idx,
        padded_input_length=seq_length,
    )

    # Generate predictions
    generated_onehot_padded = numpy.zeros((n_seqs, src.definitions.DHS64_INPUT_LENGTH, 4))
    generated_onehot_padded[:, :generated_onehot.shape[1], :] = generated_onehot
    generated_pred_design = model_design.predict(generated_onehot, verbose=1)

    # Save
    generated_design_preds_df = pandas.DataFrame(
        index=generated_seq_ids, columns=['seq'] + biosamples, 
    )
    generated_design_preds_df.index.name = 'seq_id'
    generated_design_preds_df['seq'] = generated_seqs
    generated_design_preds_df[biosamples] = generated_pred_design
    generated_design_preds_df.to_csv(
        os.path.join(output_dir, f"{output_prefix}_preds_design.csv.gz"),
        index=False,
    )

    # Generate and save predictions from validation model
    # Load validation model
    model_val = src.model.load_model(VAL_MODEL_PATH)
    model_val = src.model.select_output_head(model_val, 0)

    generated_pred_val = model_val.predict(generated_onehot_padded, verbose=1)
    
    generated_val_preds_df = pandas.DataFrame(
        index=generated_seq_ids, columns=['seq'] + biosamples, 
    )
    generated_val_preds_df.index.name = 'seq_id'
    generated_val_preds_df['seq'] = generated_seqs
    generated_val_preds_df[biosamples] = generated_pred_val
    generated_val_preds_df.to_csv(
        os.path.join(output_dir, f"{output_prefix}_preds_val.csv.gz"),
        index=False,
    )

    # Plots
    print("\nGenerating plots...")

    # Sequence bitmap
    ax = src.plot.plot_sequence_bitmap(generated_onehot)
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_seq_bitmap.png"))
    pyplot.close(fig)

    # Sample sequences
    fig = src.plot.plot_seq_logos(generated_onehot, n_seqs=10)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_seqs.png"))
    pyplot.close(fig)

    # Distribution of edit distances
    distances = src.sequence.get_paired_editdistances(generated_seqs)
    fig, ax = pyplot.subplots(1, 1, figsize=(4, 3.5))
    seaborn.violinplot(data=[distances], ax=ax)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        'Edit distance / nucleotide\n'
        '{:.3f} +/- {:.3f}'.format(numpy.mean(distances), numpy.std(distances))
    )
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_editdistance.png"))
    pyplot.close(fig)

    # Distribution of kmer distances
    try:
        distances = src.sequence.get_min_euc_nmer_dist(
            generated_seqs,
            nmer=4,
            random_seed_subsample=2020,
            normalize_counts=True,
        )
    except ValueError:
        distances = numpy.array([0]*len(generated_seqs))
    fig, ax = pyplot.subplots(1, 1, figsize=(4, 3.5))
    seaborn.violinplot(data=[distances], ax=ax)
    ax.set_xticks([])
    ax.set_ylim(0, 0.25)
    ax.set_ylabel(
        '4-mer distance\n'
        '{:.3f} +/- {:.3f}'.format(numpy.mean(distances), numpy.std(distances))
    )
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_4mer_distance.png"))
    pyplot.close(fig)

    # Design predictions across biosamples
    df_to_plot = generated_design_preds_df[biosamples].melt(
        var_name='biosample',
        value_name='prediction',
    )
    palette = {b:'lightgrey' for b in biosamples}
    palette[target] = 'tab:blue'
    fig, ax = pyplot.subplots(figsize=(9, 3.))
    seaborn.boxplot(
        data=df_to_plot,
        x='biosample',
        y='prediction',
        order=biosamples,
        hue='biosample',
        legend=False,
        palette=palette,
        fliersize=3,
        width=0.8,
        ax=ax,
    )
    ax.grid()
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    # Iterate over x axis labels and bold targets
    for label_idx, label in enumerate(ax.get_xticklabels()):
        if biosamples[label_idx]==target:
            label.set_fontweight('bold')
    ax.set_xlabel('Biosample')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nDesign model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_design_boxplot.png"))
    pyplot.close(fig)

    # Validation predictions
    df_to_plot = generated_val_preds_df[biosamples].melt(
        var_name='biosample',
        value_name='prediction',
    )
    palette = {b:'lightgrey' for b in biosamples}
    palette[target] = 'tab:blue'
    fig, ax = pyplot.subplots(figsize=(9, 3.))
    seaborn.boxplot(
        data=df_to_plot,
        x='biosample',
        y='prediction',
        order=biosamples,
        hue='biosample',
        legend=False,
        palette=palette,
        fliersize=3,
        width=0.8,
        ax=ax,
    )
    ax.grid()
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    # Iterate over x axis labels and bold targets
    for label_idx, label in enumerate(ax.get_xticklabels()):
        if biosamples[label_idx]==target:
            label.set_fontweight('bold')
    ax.set_xlabel('Biosample')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nValidation model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_val_boxplot.png"))
    pyplot.close(fig)

    print(f"\nDone with biosample {target} ({target_idx} / {len(biosamples)}).")

###############
# Entry point #
###############
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Fast SeqProp to generate sequences with biosample-specific activity using DHS64.')
    parser.add_argument('--target-idx', type=int, help='Target biosample index within all DHS64-modeled biosamples.')
    parser.add_argument('--n-seqs', type=int, default=100, help='Number of sequences to generate.')
    # parser.add_argument('--seq-length', type=int, default=145, help='Length of sequences to generate.')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save output files.')
    parser.add_argument('--output-prefix', type=str, default=None, help='Prefix for output files. If None, a prefix based on biosample index and name will be used.')
    parser.add_argument('--generator-path', type=str, default=None, help='Path to a pre-trained generator model. If None, a new generator will be trained.')
    parser.add_argument('--random-seed-train', type=int, default=None, help='Random seed for generator training.')
    parser.add_argument('--random-seed-gen', type=int, default=None, help='Random seed for sequence generation.')
    args = parser.parse_args()

    run(
        target_idx=args.target_idx,
        n_seqs=args.n_seqs,
        # seq_length=parser.seq_length,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        generator_path=args.generator_path,
        random_seed_train=args.random_seed_train,
        random_seed_gen=args.random_seed_gen
    )
