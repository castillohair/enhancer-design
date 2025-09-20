import numpy
import scipy
import matplotlib
from matplotlib import pyplot
import pandas
import seaborn

import logomaker

# Nucleotide colors
NT_COLOR_DICT = {
    'A': (15/255, 148/255, 71/255),
    'C': (35/255, 63/255, 153/255),
    'G': (245/255, 179/255, 40/255),
    'T': (228/255, 38/255, 56/255),
}

def plot_sequence_bitmap(seq_vals, ax=None, legend=True):
    """
    Plot a list of sequences as a colormap, with each sequence in its own row.

    Parameters
    ----------
    seq_vals : list of str or numpy.ndarray
        List of sequences to plot. Could be raw strings or one hot-encoded.
    ax : matplotlib.Axes or None
        Axes to plot on. If None, create a new figure.
    legend : bool
        Whether to include a legend for nucleotide colors.

    Returns
    -------
    matplotlib.Axes
        The Axes object with the plot.

    """
    # Convert sequences to numerical indices
    if type(seq_vals[0])==numpy.ndarray:
        # Assume one hot-encoded
        seqs_as_index = numpy.argmax(seq_vals, axis=-1)
    elif type(seq_vals[0])==str:
        # Assume list of strings
        seqs_as_index = [[["A", "C", "G", "T"].index(c) for c in si] for si in seq_vals]
        seqs_as_index = numpy.array(seqs_as_index)
    else:
        raise ValueError(f"type of seq_vals {type(seq_vals)} not recognized")
            
    # Define colors and colormap
    nt_colors = [NT_COLOR_DICT[n] for n in ['A', 'C', 'G', 'T']]
    cmap = matplotlib.colors.ListedColormap(nt_colors)
    bounds=[0, 1, 2, 3, 4]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Actually plot
    if ax is None:
        fig, ax = pyplot.subplots(figsize=(0.03*seqs_as_index.shape[1], 0.03*seqs_as_index.shape[0]))
    else:
        fig = ax.figure
    ax.imshow(
        seqs_as_index[::-1] + 0.5,
        aspect='equal',
        interpolation='nearest',
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    # Custom legend with nucleotide colors
    if legend:
        legend_elements = [
            matplotlib.patches.Patch(facecolor=c, label=nt)
            for nt, c in NT_COLOR_DICT.items()
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.015), loc='upper left', fontsize='medium')

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel('Position (nt)')
    ax.set_ylabel('Generated sequences')

    return ax

def plot_seq_logo(nt_height=None, pwm=None, seq=None, font_name='DejaVu Sans Mono', ax=None, title=None):
    """
    Plot a sequence logo
    
    Input can be specified as a sequence string, pwm matrix, or nucleotide height matrix.

    Parameters
    ----------
    nt_height : numpy.ndarray or None
        Nucleotide height matrix.
    pwm : numpy.ndarray or None
        PWM matrix.
    seq : str or None
        Sequence string.
    font_name : str
        Font name for the logo text.
    ax : matplotlib.Axes or None
        Axes to plot on.
    title : str or None
        Title for the plot.

    Returns
    -------
    matplotlib.Axes
        Axes containing the sequence logo.

    """

    if nt_height is None and seq is None and pwm is None:
        raise ValueError("At least one of nt_height, seq, or pwm must be provided")
    
    # Preference is given to nt_height, then pwm, then seq
    if nt_height is None:
        if pwm is not None:
            # Infer nucleotide heights using information content / entropy
            entropy = numpy.zeros_like(pwm)
            entropy[pwm > 0] = pwm[pwm > 0] * -numpy.log2(pwm[pwm > 0])
            entropy = numpy.sum(entropy, axis=1)
            conservation = 2 - entropy
            # Nucleotide height
            nt_height = numpy.tile(numpy.reshape(conservation, (-1, 1)), (1, 4))
            nt_height = pwm * nt_height
        elif seq is not None:
            # Nucleotide heights from one hot-encoding of sequence
            nt_to_onehot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
            nt_height = [nt_to_onehot[c] for c in seq.upper()]
            nt_height = numpy.array(nt_height)

    nt_height_df = pandas.DataFrame(
        nt_height,
        columns=['A', 'C', 'G', 'T'],
    )

    if ax is None:
        fig, ax = pyplot.subplots(figsize=(len(nt_height_df)/20, 0.5))
    
    logo = logomaker.Logo(
        nt_height_df,
        color_scheme=NT_COLOR_DICT,
        ax=ax,
        font_name=font_name,
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True, linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    return ax

def plot_seq_logos(seq_vals, n_seqs=None):
    """
    Plot sequence logos for a list of sequences.

    Parameters
    ----------
    seq_vals : list
        List of sequence strings or nucleotide height matrices.
    n_seqs : int, optional
        Number of sequences to plot. If None, plot all sequences.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the sequence logos.

    """
    if n_seqs is None:
        n_seqs = len(seq_vals)
    else:
        n_seqs = min(len(seq_vals), n_seqs)
    seq_len = len(seq_vals[0])

    fig, axes = pyplot.subplots(n_seqs, 1, figsize=(seq_len/10, 0.4*n_seqs))
    for seq_idx in range(n_seqs):
        if isinstance(seq_vals[seq_idx], str):
            plot_seq_logo(seq=seq_vals[seq_idx], ax=axes[seq_idx])
        else:
            plot_seq_logo(nt_height=seq_vals[seq_idx], ax=axes[seq_idx])

    return fig
