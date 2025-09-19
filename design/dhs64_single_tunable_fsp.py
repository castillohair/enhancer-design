"""
Generate DNA sequences with chromatin accessibility specific to a target biosample, using
the DHS64 model and Fast SeqProp as the optimization method. Instead of maximizing predicted
target accessibility, generated sequences are optimized to achieve a range of target
accessibility setpoints linearly spaced between a specified minimum and maximum value.

Script sections:
1. Imports and constants: Import necessary libraries and define paths to model files and
   metadata. Additionally, define default minimum and maximum setpoints for each biosample.
   These were obtained by running Fast SeqProp to generate sequences that maximize target
   biosample accessibility and minimize non-target accessibility, and recording average
   predictions across sequences for each biosample and target. The default minimum was obtained
   from the minimum average prediction that a biosample reached across design targets, and the
   default maximum was obtained as 1.5 times the maximum average value.
2. Loss functions: Specify sequence features to optimize. We define a target loss function
   that minimizes the distance of the target biosample prediction to the desired setpoint value,
   and minimizes non-target biosample predictions. Additionally, we define a PWM loss function
   that penalizes repeated nucleotides, extreme GC content, and forbidden sequences that include
   restriction enzyme sites. This avoids the need for later removing generated sequences, which
   would leave holes in the setpoint distribution.
3. Main sequence design function: Loads models and metadata, runs Fast SeqProp to generate
   sequences, calculates predictions, saves results, and generates plots. Note that we use
   an ensemble of models for design, taking the average prediction across the target biosample,
   and the maximum prediction across non-target biosamples. We additionally generate predictions
   from a separate validation model.
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

# Default setpoints for each biosample
DEFAULT_MIN_SETPOINTS = numpy.array(
    [
        -1.8759682, -1.8494011, -1.8569218, -2.0512285, -1.8080542,
        -1.8286018, -1.6516305, -1.8802142, -1.7351601, -1.9402835,
        -1.9480613, -1.9012293, -1.8837895, -1.9739637, -1.9936752,
        -1.9600551, -2.041561 , -1.6031964, -1.9650493, -1.9019052,
        -1.5624664, -2.2934916, -1.9236313, -1.7366997, -1.6660352,
        -1.8184152, -1.9587451, -2.2249029, -2.2409315, -1.9760346,
        -1.6236752, -1.5731835, -1.8095546, -1.8582834, -2.1956942,
        -1.8359246, -2.0465255, -1.8122882, -1.709302 , -1.7798306,
        -1.7296495, -1.9407419, -2.0812936, -2.1493614, -2.1099331,
        -1.9190738, -2.0146537, -1.8489852, -1.7334645, -1.8975159,
        -1.7342676, -1.8385947, -1.8728017, -2.1823466, -1.6864258,
        -1.7721314, -1.7864283, -1.7685765, -1.8832142, -1.9623082,
        -1.7230127, -1.9109275, -1.8265659, -1.7982031,
    ]
)
DEFAULT_MAX_SETPOINTS = numpy.array(
    [
        1.389358  ,  1.9062928 ,  0.6517941 ,  1.4612527 ,  2.5909956 ,
        2.0436518 ,  3.0010138 ,  1.763699  , -0.3199729 , -0.13173959,
        0.35857636,  1.1704576 ,  2.1626823 ,  1.5840851 ,  1.7921948 ,
        3.2213087 ,  1.1231854 ,  2.7105799 ,  2.6105886 ,  2.365243  ,
        1.5554183 ,  2.1670263 ,  0.6409646 ,  1.2298881 ,  3.1545882 ,
        2.6453772 ,  2.0458624 ,  1.9625995 ,  2.7884939 ,  5.02283   ,
        3.2326534 ,  3.7848306 ,  2.3998418 ,  2.2659311 ,  2.356655  ,
        2.0067527 ,  2.7032092 ,  1.747683  ,  0.66412693,  2.8326528 ,
        2.5775983 ,  2.5194192 ,  1.8466012 ,  0.9591188 ,  1.9650254 ,
        4.176268  ,  2.9436803 ,  4.021358  ,  3.1799762 ,  4.6714373 ,
        1.5532568 ,  1.301442  ,  1.151238  ,  0.6185183 ,  2.7086868 ,
        2.7175403 ,  4.336569  ,  1.7777421 ,  1.8546844 ,  2.02806   ,
        3.773029  ,  2.7578254 ,  2.9910457 ,  0.80838096,
    ]
)

##################
# Loss functions #
##################
def get_target_setpoint_loss_func(
        target_idx,
        target_setpoint_vals,
        n_model_outputs,
        target_weight,
        non_target_weight,
    ):
    """
    Get a target loss function to provide to Fast SeqProp.
    
    The returned function minimizes the distance of the target biosample prediction
    to the desired setpoint value(s), and minimizes the predictions of all other
    biosamples.

    Parameters
    ----------
    target_idx : int
        Index of the target biosample to maximize.
    target_setpoint_vals : float or array-like
        Desired setpoint value(s) for the target biosample prediction. If an array,
        it should have shape (n_seqs,).
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
    target_setpoint_vals = tensorflow.cast(target_setpoint_vals, tensorflow.float32)
    target_mask = numpy.zeros((1, n_model_outputs))
    target_mask[:, target_idx] = 1
    target_mask = tensorflow.cast(target_mask, tensorflow.float32)

    def target_loss_func(model_preds):
        # model_preds has dimensions (n_seqs, n_outputs)
        target_score = tensorflow.reduce_mean(
            tensorflow.math.abs(model_preds[:, target_idx] - target_setpoint_vals)
        )
        non_target_score = tensorflow.reduce_mean((1 - target_mask)*model_preds)
        return non_target_weight*non_target_score + target_weight*target_score
    
    return target_loss_func

# The following functions are used for the PWM loss function

# The following penalizes repeats
def repeat_loss_func(pwm):
    # PWM has dimensions (n_seqs, seq_length, n_channels)
    return tensorflow.reduce_mean(pwm[:, :-1, :] * pwm[:, 1:, :])

# The following penalizes extreme GC contents
def gc_loss_func(pwm):
    return tensorflow.reduce_mean(
        tensorflow.reduce_mean(
            pwm[:, :, 1] + pwm[:, :, 2] - pwm[:, :, 0] - pwm[:, :, 3],
            axis=-1,
        )**2
    )

# The following penalizes forbidden sequences
def forbidden_sites_loss_func(pwm):
    # KpnI: GGTACC
    kpni_penalty = tensorflow.reduce_mean(
        pwm[:, :-5, 2]*pwm[:, 1:-4, 2]*pwm[:, 2:-3, 3]*pwm[:, 3:-2, 0]*pwm[:, 4:-1, 1]*pwm[:, 5:, 1]
    )
    # XbaI: TCTAGA
    xbai_penalty = tensorflow.reduce_mean(
        pwm[:, :-5, 3]*pwm[:, 1:-4, 1]*pwm[:, 2:-3, 3]*pwm[:, 3:-2, 0]*pwm[:, 4:-1, 2]*pwm[:, 5:, 0]
    )
    # Partial KpnI: start with GTACC
    partial_kpni_penalty = tensorflow.reduce_mean(
        pwm[:, 0, 2]*pwm[:, 1, 3]*pwm[:, 2, 0]*pwm[:, 3, 1]*pwm[:, 4, 1]
    )
    return kpni_penalty + xbai_penalty + partial_kpni_penalty

def get_pwm_loss_func(
        repeat_loss_weight=1.0,
        gc_loss_weight=0.0,
        forbidden_sites_loss_weight=0.0,
    ):
    """
    Get a PWM loss function to provide to Fast SeqProp.

    The returned function penalizes repeated nucleotides, extreme GC content,
    and forbidden sequences in the PWM.

    Parameters
    ----------
    repeat_loss_weight : float
        Weight for the repeat loss component.
    gc_loss_weight : float
        Weight for the GC content loss component.
    forbidden_sites_loss_weight : float
        Weight for the forbidden sites loss component.

    Returns
    -------
    function
        Loss function.

    """
    def pwm_loss_func(pwm):
        # PWM has dimensions (n_seqs, seq_length, n_channels)
        return repeat_loss_weight*repeat_loss_func(pwm) + \
            gc_loss_weight*gc_loss_func(pwm) + \
            forbidden_sites_loss_weight*forbidden_sites_loss_func(pwm)
    
    return pwm_loss_func

#################################
# Main sequence design function #
#################################
def run(
        target_idx,
        setpoint_vals,
        seq_length,
        n_seqs,
        output_dir='.',
        output_prefix=None,
        seed=None,
    ):
    """
    Run Fast SeqProp to generate sequences with tunable biosample-specific activity using DHS64.

    Parameters
    ----------
    target_idx : int
        Index of the biosample to target within all DHS64-modeled biosamples.
    setpoint_vals : float or array-like
        Desired setpoint value(s) for the target biosample prediction. If an array,
        it should have shape (n_seqs,).
    seq_length : int
        Length of sequences to generate.
    n_seqs : int
        Number of sequences to generate.
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
    biosample_metadata_df = pandas.read_excel(BIOSAMPLE_META_PATH)
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
            'pwm_weight': 1,
            'entropy_weight': 1e-3,
            'learning_rate': 0.001,
            'n_iter_max': 2500,
            'init_seed': seed,
        },
        'target_loss_params': {
            'target_idx': target_idx,
            'target_setpoint_vals': setpoint_vals.tolist() if not numpy.isscalar(setpoint_vals) else float(setpoint_vals),
            'n_model_outputs': len(biosamples),
            'target_weight': 1,
            'non_target_weight': 1,
        },
        'pwm_loss_params': {
            'repeat_loss_weight': 4.0,
            'gc_loss_weight': 1.0,
            'forbidden_sites_loss_weight': 1.0,
        },
    }
    # Save run parameters
    with open(os.path.join(output_dir, f'{output_prefix}_run_metadata.json'), 'w') as file:
        file.write(json.dumps(run_parameters, indent=4))

    # Get loss functions
    target_loss_func = get_target_setpoint_loss_func(
        **run_parameters['target_loss_params'],
    )
    pwm_loss_func = get_pwm_loss_func(**run_parameters['pwm_loss_params'])

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
    model_val = src.model.select_output_head(model_val, 0)
    generated_onehot_padded = numpy.zeros((n_seqs, src.definitions.DHS64_INPUT_LENGTH, 4), dtype=numpy.float32)
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
    palette[target] = 'tab:red'
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
    # Horizontal line at setpoint if single value
    if numpy.isscalar(setpoint_vals):
        ax.axhline(setpoint_vals, color='tab:red', linestyle='--', label='Setpoint', zorder=-1)
    # Iterate over x axis labels and bold targets
    for label_idx, label in enumerate(ax.get_xticklabels()):
        if biosamples[label_idx]==target:
            label.set_fontweight('bold')
    ax.set_xlabel('Biosample')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nDesign model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_design_boxplot.png"))
    pyplot.close(fig)

    # Scatter of setpoint vs. design prediction for target biosample if setpoint is an array
    if not numpy.isscalar(setpoint_vals):
        fig, ax = pyplot.subplots(figsize=(4, 4))
        ax.scatter(setpoint_vals, generated_design_preds_df[target], s=5)
        ax.axline(
            (numpy.mean(setpoint_vals), numpy.mean(setpoint_vals)),
            slope=1,
            linestyle='-',
            label='y=x',
            zorder=-1,
        )
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('Setpoint')
        ax.set_ylabel('Design model prediction')
        ax.set_title(f'Target biosample: {target}')
        fig.savefig(os.path.join(output_dir, f"{output_prefix}_setpoint_vs_preds_design_scatter.png"))
        pyplot.close(fig)

    # Validation predictions
    df_to_plot = generated_val_preds_df[biosamples].melt(
        var_name='biosample',
        value_name='prediction',
    )
    palette = {b:'lightgrey' for b in biosamples}
    palette[target] = 'tab:red'
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
    # Horizontal line at setpoint if single value
    if numpy.isscalar(setpoint_vals):
        ax.axhline(setpoint_vals, color='tab:red', linestyle='--', label='Setpoint', zorder=-1)
    # Iterate over x axis labels and bold targets
    for label_idx, label in enumerate(ax.get_xticklabels()):
        if biosamples[label_idx]==target:
            label.set_fontweight('bold')
    ax.set_xlabel('Biosample')
    ax.set_ylabel('$log_{10}$ accessibility prediction\nValidation model')
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_preds_val_boxplot.png"))
    pyplot.close(fig)

    # Scatter of setpoint vs. design prediction for target biosample if setpoint is an array
    if not numpy.isscalar(setpoint_vals):
        fig, ax = pyplot.subplots(figsize=(4, 4))
        ax.scatter(setpoint_vals, generated_val_preds_df[target], s=5)
        ax.axline(
            (numpy.mean(setpoint_vals), numpy.mean(setpoint_vals)),
            slope=1,
            linestyle='-',
            label='y=x',
            zorder=-1,
        )
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('Setpoint')
        ax.set_ylabel('Validation model prediction')
        ax.set_title(f'Target biosample: {target}')
        fig.savefig(os.path.join(output_dir, f"{output_prefix}_setpoint_vs_preds_val_scatter.png"))
        pyplot.close(fig)

    print(f"\nDone with biosample {target} ({target_idx} / {len(biosamples)}).")

###############
# Entry point #
###############
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Fast SeqProp to generate sequences with biosample-specific activity using DHS64.')
    parser.add_argument('--target-idx', type=int, help='Target biosample index within all DHS64-modeled biosamples.')
    parser.add_argument('--setpoint-min', type=float, default=None, help='Minimum setpoint value for the target biosample. If not specified, a default value will be used.')
    parser.add_argument('--setpoint-max', type=float, default=None, help='Maximum setpoint value for the target biosample. If not specified, a default value will be used.')
    parser.add_argument('--seq-length', type=int, default=145, help='Length of sequences to generate.')
    parser.add_argument('--n-seqs', type=int, default=100, help='Number of sequences to generate.')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save output files.')
    parser.add_argument('--output-prefix', type=str, default=None, help='Prefix for output files. If None, a prefix based on biosample index and name will be used.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sequence initialization. If None, a random seed will be used.')
    args = parser.parse_args()

    # Construct setpoint values array
    if args.setpoint_min is None:
        setpoint_min = DEFAULT_MIN_SETPOINTS[args.target_idx]
    else:
        setpoint_min = args.setpoint_min
    if args.setpoint_max is None:
        setpoint_max = DEFAULT_MAX_SETPOINTS[args.target_idx]
    else:
        setpoint_max = args.setpoint_max
    setpoint_vals = numpy.linspace(setpoint_min, setpoint_max, args.n_seqs)

    run(
        target_idx=args.target_idx,
        setpoint_vals=setpoint_vals,
        seq_length=args.seq_length,
        n_seqs=args.n_seqs,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        seed=args.seed,
    )
