"""
Generate DNA sequences with chromatin accessibility specific to a target biosample, using
the DH733 model and Fast SeqProp as the optimization method.

Script sections:
1. Imports and constants: Import necessary libraries and define paths to model files and
   metadata.
2. Loss functions: Specify sequence features to optimize. We define a target loss function
   that maximizes target biosample accessibility and minimizes a specified percentile of
   non-target biosample predictions. Higher non-target percentile corresponds to more
   stringent designs but may lead to lower target activity. Additionally, we define a PWM
   loss function that penalizes repeated nucleotides.
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
import tensorflow_probability as tfp

import corefsp

BASE_DIR = '../'
sys.path.append(BASE_DIR)
import src.definitions
import src.model
import src.sequence
import src.plot
import src.utils

BIOSAMPLE_META_PATH = 'dhs733_nonredundant_biosample_metadata.tsv'
OUTPUT_TRANSFORMATION_MATRIX_PATH = 'dhs733_nonredundant_transformation_matrix.npy'
DESIGN_MODEL_PATHS = [os.path.join(BASE_DIR, src.definitions.DHS733_MODEL_PATH[i]) for i in [1, 3]]
VAL_MODEL_PATH = os.path.join(BASE_DIR, src.definitions.DHS733_MODEL_PATH[0])

##################
# Loss functions #
##################
def get_target_percentile_loss_func(
        target_idx,
        n_model_outputs,
        target_weight,
        non_target_weight,
        non_target_percentile,
    ):
    """
    Get a target loss function to provide to Fast SeqProp.
    
    The returned function maximizes the predicted difference between target biosample
    and a specified percentile of non-target biosample predictions.

    Parameters
    ----------
    target_idx : int
        Index of the target biosample to maximize.
    n_model_outputs : int
        Number of outputs of the model.
    target_weight : float
        Weight for the target biosample score in the loss.
    non_target_weight : float
        Weight for the non-target biosample score in the loss.
    non_target_percentile : float
        Percentile of non-target biosample predictions to explicitly minimize.

    Returns
    -------
    function
        Loss function.

    """
    nontarget_biosample_idx = [i for i in range(n_model_outputs) if i!=target_idx]
    nontarget_biosample_idx = tensorflow.cast(nontarget_biosample_idx, tensorflow.int32)

    def target_loss_func(model_preds):
        # model_preds has dimensions (n_seqs, n_outputs)
        model_preds_target = model_preds[:, target_idx]
        model_preds_nontarget = tensorflow.gather(
            model_preds,
            nontarget_biosample_idx,
            axis=1,
        )
        target_score = - tensorflow.reduce_mean(model_preds_target)
        non_target_score = tensorflow.reduce_mean(
            tfp.stats.percentile(model_preds_nontarget, non_target_percentile, interpolation='midpoint', axis=1)
        )

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
        target_idx,
        seq_length,
        n_seqs,
        non_target_percentile=95,
        output_dir='.',
        output_prefix=None,
        seed=None,
    ):
    """
    Run Fast SeqProp to generate sequences with biosample-specific activity using DHS733.

    Parameters
    ----------
    target_idx : int
        Index of the biosample to target within non-redundant DHS733-modeled biosamples.
    seq_length : int
        Length of sequences to generate.
    n_seqs : int
        Number of sequences to generate.
    non_target_percentile : float, optional
        Percentile of non-target biosample predictions to explicitly minimize. The higher
        this value, the more stringent the specificity may be, but target activity may
        be lower. Default is 95.
    output_dir : str, optional
        Directory to save output files. Default is current directory.
    output_prefix : str, optional
        Prefix for output files. If None, a prefix based on biosample index and name will be used.
    seed : int, optional
        Random seed for sequence initialization. If None, a random seed will be used.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load metadata of modeled biosamples
    biosample_metadata_df = pandas.read_csv(BIOSAMPLE_META_PATH, sep='\t')
    biosamples = biosample_metadata_df['Biosample name'].tolist()

    # Extract and sanitize biosample name
    target = biosamples[target_idx]
    target_sanitized = src.utils.sanitize_str(target)

    # Prefix for output files
    if output_prefix is None:
        output_prefix = f"{target_idx}_{target_sanitized}"

    print(f"\nStarting sequence design for biosample {target} ({target_idx} / {len(biosamples)})...")

    # Construct model for design
    # ==========================
    print("\nLoading design model...")

    # Load models to be used for design
    models_design_list = [src.model.load_model(filepath) for filepath in DESIGN_MODEL_PATHS]
    # Transform model output
    models_design_list = [src.model.apply_output_transformation(m, OUTPUT_TRANSFORMATION_MATRIX_PATH) for m in models_design_list]

    # Pessimistic ensemble: minimum across target biosample, maximum across non-target biosamples
    min_output_idx = [target_idx]
    max_output_idx = [i for i in range(len(biosamples)) if i != target_idx]
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
        'target_idx': target_idx,
        'fsp_params': {
            'seq_length': seq_length,
            'n_seqs': n_seqs,
            'target_weight': 1,
            'pwm_weight': 0.3,
            'entropy_weight': 1e-3,
            'learning_rate': 0.001,
            'n_iter_max': 10000,
            'init_seed': seed,
            'early_stopping': True,
            'early_stopping_mode': 'rel',
            'early_stopping_min_delta': 0.001,
            'early_stopping_min_batch': 50,
        },
        'target_loss_params': {
            'target_idx': target_idx,
            'n_model_outputs': len(biosamples),
            'target_weight': 1,
            'non_target_weight': 1,
            'non_target_percentile': non_target_percentile,
        },
        'pwm_loss_params': {
        },
    }
    # Save run parameters
    with open(os.path.join(output_dir, f'{output_prefix}_run_metadata.json'), 'w') as file:
        file.write(json.dumps(run_parameters, indent=4))

    # Get loss functions
    target_loss_func = get_target_percentile_loss_func(
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
    generated_design_preds_df = generated_df.copy()
    generated_design_preds_df[biosamples] = generated_pred_design
    generated_design_preds_df.to_csv(
        os.path.join(output_dir, f"{output_prefix}_preds_design.csv.gz"),
        index=False,
    )
    
    # Generate predictions from validation model
    print("Generating predictions from validation model...")
    model_val = src.model.load_model(VAL_MODEL_PATH)
    model_val = src.model.apply_output_transformation(model_val, OUTPUT_TRANSFORMATION_MATRIX_PATH)
    generated_onehot_padded = numpy.zeros((n_seqs, src.definitions.DHS733_INPUT_LENGTH, 4), dtype=numpy.float32)
    generated_onehot_padded[:, :generated_onehot.shape[1], :] = generated_onehot
    generated_pred_val = model_val.predict(generated_onehot_padded, verbose=1)
    
    generated_val_preds_df = generated_df.copy()
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
    palette[target] = 'tab:blue'
    fig, ax = pyplot.subplots(figsize=(20, 3.5))
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
    ax.tick_params(axis='x', rotation=90, labelsize=4.5)
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
    fig, ax = pyplot.subplots(figsize=(20, 3.5))
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
    ax.tick_params(axis='x', rotation=90, labelsize=4.5)
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

    parser = argparse.ArgumentParser(description='Run Fast SeqProp to generate sequences with biosample-specific activity using DHS733.')
    parser.add_argument('--target-idx', type=int, help='Target biosample index within non-redundant DHS733-modeled biosamples. See "dhs733_nonredundant_biosample_metadata.tsv" for a list of possible target biosamples.')
    parser.add_argument('--seq-length', type=int, default=145, help='Length of sequences to generate.')
    parser.add_argument('--n-seqs', type=int, default=100, help='Number of sequences to generate.')
    parser.add_argument('--non-target-percentile', type=float, default=95, help='Percentile of non-target biosample predictions to explicitly minimize. Higher non-target percentile corresponds to more stringent designs.')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save output files.')
    parser.add_argument('--output-prefix', type=str, default=None, help='Prefix for output files. If None, a prefix based on biosample index and name will be used.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sequence initialization. If None, a random seed will be used.')
    args = parser.parse_args()

    run(
        target_idx=args.target_idx,
        seq_length=args.seq_length,
        n_seqs=args.n_seqs,
        non_target_percentile=args.non_target_percentile,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        seed=args.seed,
    )
