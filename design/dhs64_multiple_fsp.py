"""
Generate DNA sequences with chromatin accessibility specific to multiple biosamples, using
the DHS64 model and Fast SeqProp as the optimization method.

Script sections:
1. Imports and constants: Import necessary libraries and define paths to model files and
   metadata.
2. Loss functions: Specify sequence features to optimize. We define a target loss function
   that maximizes both the minimum and average predictions across target biosamples, and
   minimizes average non-target predictions. Additionally, we define a PWM loss function
   that penalizes repeated nucleotides.
3. Main sequence design function: Loads models and metadata, runs Fast SeqProp to generate
   sequences, calculates predictions, saves results, and generates plots. Note that we use
   a pessimistic ensemble of models for design, taking the minimum prediction across the
   target biosample and the maximum prediction across non-target biosamples. We additionally
   generate predictions from a separate validation model.
4. Entry point: Parses command-line arguments and runs the main design function.

"""

import argparse
import datetime
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

import corefsp

BASE_DIR = '../'
sys.path.append(BASE_DIR)
import src.definitions
import src.model
import src.sequence
import src.plot
import src.utils

BIOSAMPLE_META_PATH = os.path.join(BASE_DIR, src.definitions.DHS64_BIOSAMPLE_META_PATH)
DESIGN_MODEL_PATHS = [os.path.join(BASE_DIR, src.definitions.DHS64_MODEL_PATH[i]) for i in [1, 3]]
VAL_MODEL_PATH = os.path.join(BASE_DIR, src.definitions.DHS64_MODEL_PATH[0])

##################
# Loss functions #
##################
def get_target_min_plus_avg_loss_func(
        targets_idx,
        n_model_outputs,
        target_weight,
        non_target_weight,
    ):
    """
    Get a target loss function to provide to Fast SeqProp.
    
    The returned function maximizes a weighted sum of the minimum and the average target
    biosample predictions, and minimizes the average non-target biosample predictions.

    Parameters
    ----------
    targets_idx : int
        List of indices of biosamples to maximize.
    n_model_outputs : int
        Total number of model outputs (biosamples).
    target_weight : float
        Weight for the target biosample score in the loss.
    non_target_weight : float
        Weight for the non-target biosample score in the loss.

    Returns
    -------
    function
        Loss function.

    """

    nontargets_idx = [i for i in range(n_model_outputs) if i not in targets_idx]
    targets_idx = tensorflow.cast(targets_idx, tensorflow.int32)
    nontargets_idx = tensorflow.cast(nontargets_idx, tensorflow.int32)

    def target_loss_func(model_preds):
        # model_preds has dimensions (n_seqs, n_outputs)
        n_model_outputs = model_preds.shape[1]
        model_preds_target = tensorflow.gather(
            model_preds,
            targets_idx,
            axis=1,
        )
        model_preds_nontarget = tensorflow.gather(
            model_preds,
            nontargets_idx,
            axis=1,
        )
        
        # Combination of min and average
        target_score_min = - tensorflow.reduce_mean(
            tensorflow.reduce_min(model_preds_target, axis=1)
        )
        target_score_avg = - tensorflow.reduce_mean(
            tensorflow.reduce_mean(model_preds_target, axis=1)
        )
        target_score = target_score_min + 0.2*target_score_avg

        non_target_score = tensorflow.reduce_mean(model_preds_nontarget)

        return non_target_weight*non_target_score + target_weight*target_score
        
    return target_loss_func

def get_repeat_loss_func():
    """
    Get a PWM loss function to provide to Fast SeqProp.

    The returned function penalizes repeated nucleotides in the PWM.

    Returns
    -------
    function
        Loss function.

    """
    def repeat_loss_func(pwm):
        # PWM has dimensions (n_seqs, seq_length, n_channels)
        return tensorflow.reduce_mean(pwm[:, :-1, :] * pwm[:, 1:, :])
    
    return repeat_loss_func

#################################
# Main sequence design function #
#################################
def run(
        targets_idx,
        n_seqs,
        seq_length,
        output_dir='.',
        output_prefix=None,
        seed=None,
    ):
    """
    Run Fast SeqProp to generate sequences with activity specific to multiple biosamples using DHS64 model.
    
    Parameters
    ----------
    targets_idx : list of int
        List of indices of biosamples to maximize among all DHS64-modeled biosamples.
    n_seqs : int
        Number of sequences to generate.
    seq_length : int
        Length of sequences to generate.
    output_dir : str, optional
        Directory to save output files. Default is current directory.
    output_prefix : str, optional
        Prefix for output files. If None, a prefix based on target indices and names will be used.
    seed : int, optional
        Random seed for sequence initialization. If None, a random seed will be used.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load metadata of modeled biosamples
    biosample_metadata_df = pandas.read_excel(BIOSAMPLE_META_PATH)
    biosamples = biosample_metadata_df['Biosample name'].tolist()

    # Extract and sanitize biosample name
    targets = [biosamples[i] for i in targets_idx]
    targets_sanitized = [src.utils.sanitize_str(b) for b in targets]

    # Prefix for output files
    if output_prefix is None:
        output_prefix = '_'.join([f"{i}_{n}" for i, n in zip(targets_idx, targets_sanitized)])

    print(
        f"\nStarting sequence design for targets "
        f"{', '.join([f'{n} ({i} / {len(biosamples)})' for i, n in zip(targets_idx, targets)])}..."
    )

    # Construct model for design
    # ==========================
    print("\nLoading design model...")

    # Load models to be used for design
    models_design_list = [src.model.load_model(filepath) for filepath in DESIGN_MODEL_PATHS]
    # Select first output head: continous accessibility prediction
    models_design_list = [src.model.select_output_head(m, 0) for m in models_design_list]

    # Pessimistic ensemble: minimum across target biosamples, maximum across non-target biosamples
    min_output_idx = targets_idx
    max_output_idx = [i for i in range(len(biosamples)) if i not in targets_idx]
    model_design = src.model.make_model_ensemble(
        models_design_list,
        min_output_idx=min_output_idx,
        max_output_idx=max_output_idx,
        padded_input_length=seq_length,
    )

    # Generate sequences
    # ==================
    print("\nGenerating sequences...")

    # Run parameters
    run_parameters = {
        'run_id': datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        'targets_idx': targets_idx,
        'fsp_params': {
            'seq_length': seq_length,
            'n_seqs': n_seqs,
            'target_weight': 1,
            'pwm_weight': 3,
            'entropy_weight': 1e-3,
            'learning_rate': 0.001,
            'n_iter_max': 2500,
            'init_seed': seed,
        },
        'target_loss_params': {
            'targets_idx': targets_idx,
            'n_model_outputs': len(biosamples),
            'target_weight': 1,
            'non_target_weight': 1,
        },
        'pwm_loss_params': {
        },
    }
    # Save run parameters
    with open(os.path.join(output_dir, f'{output_prefix}_run_metadata.json'), 'w') as file:
        file.write(json.dumps(run_parameters, indent=4))

    # Get loss functions
    target_loss_func = get_target_min_plus_avg_loss_func(
        **run_parameters['target_loss_params'],
    )
    pwm_loss_func = get_repeat_loss_func(**run_parameters['pwm_loss_params'])

    # Run Fast SeqProp
    generated_onehot, generated_pred_design, train_history = corefsp.design_seqs(
        model_design,
        target_loss_func=target_loss_func,
        pwm_loss_func=pwm_loss_func,
        **run_parameters['fsp_params'],
    )

    # Save results
    # ============
    print("\nSaving results...")

    # Save sequences as fasta
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

    # Save predictions from design model
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
    
    # Generate predictions from validation model
    print("\nGenerating predictions from validation model...")
    model_val = src.model.load_model(VAL_MODEL_PATH)
    model_val = src.model.select_output_head(model_val, 0)
    generated_onehot_padded = numpy.zeros((n_seqs, src.definitions.DHS64_INPUT_LENGTH, 4), dtype=numpy.float32)
    generated_onehot_padded[:, :generated_onehot.shape[1], :] = generated_onehot
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

    # Training history
    fig, axes = pyplot.subplots(1, len(train_history), figsize=(4*len(train_history), 3))
    for ax, (loss_component_key, loss_component_val) in zip(axes, train_history.items()):
        ax.plot(loss_component_val)
        ax.set_title(loss_component_key.replace('_', ' ').capitalize())
        ax.set_xlabel("Iteration")
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_train_history.png"))
    pyplot.close(fig)
    
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
    for biosample in targets:
        palette[biosample] = 'tab:blue'
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
        if biosamples[label_idx] in targets:
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
    for biosample in targets:
        palette[biosample] = 'tab:blue'
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
        if biosamples[label_idx] in targets:
            label.set_fontweight('bold')
    ax.set_xlabel('Biosample')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nValidation model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_val_boxplot.png"))
    pyplot.close(fig)
    print(
        f"\nDone with targets "
        f"{', '.join([f'{n} ({i} / {len(biosamples)})' for i, n in zip(targets_idx, targets)])}..."
    )

###############
# Entry point #
###############
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Fast SeqProp to generate sequences with activity spcecific to multiple biosamples using DHS64.')
    parser.add_argument('--targets-idx', type=str, help='Indices of biosamples to maximize among all DHS64-modeled biosamples, as a comma-separated list.', required=True)
    parser.add_argument('--n-seqs', type=int, default=100, help='Number of sequences to generate.')
    parser.add_argument('--seq-length', type=int, default=145, help='Length of sequences to generate.')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save output files.')
    parser.add_argument('--output-prefix', type=str, default=None, help='Prefix for output files. If None, a prefix based on target biosample indices and names will be used.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sequence initialization. If None, a random seed will be used.')
    args = parser.parse_args()

    # Process biosample indices
    try:
        targets_idx = [int(idx) for idx in args.targets_idx.split(',')]
    except Exception as e:
        raise ValueError("Error parsing --biosamples-idx argument. Please provide a comma-separated list of integers.")

    # Run design
    run(
        targets_idx=targets_idx,
        n_seqs=args.n_seqs,
        seq_length=args.seq_length,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        seed=args.seed,
    )
