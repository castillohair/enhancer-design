import pandas

from . import definitions

def load_data(
        file_path=definitions.MPRA_DATA_PATH,
        split_multitarget=True,
    ):
    """
    Load MPRA data and do basic preprocessing.

    Parameters
    ----------
    file_path : str, optional
        Path to the TSV file containing MPRA data.
    split_multitarget : bool, optional
        Whether to split values in the 'target' column into tuples if they
        correspond to multiple targets.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the loaded MPRA data.
    
    """
    # Load
    mpra_df = pandas.read_csv(file_path, sep='\t', index_col=0)

    # Split multiple targets into tuples
    if split_multitarget:
        get_targets_as_tuple = lambda df: df['target'].str.strip().str.split(', ').apply(lambda x: tuple(sorted(x)))
        
        mpra_double_target_df = mpra_df[
            mpra_df['category'].isin(['double_genomic', 'double_fsp', 'double_motif_embedding'])
        ]
        mpra_df.loc[mpra_double_target_df.index, 'target'] = get_targets_as_tuple(mpra_double_target_df)

        mpra_triple_target_df = mpra_df[
            mpra_df['category'].isin(['triple_genomic', 'triple_fsp', 'triple_motif_embedding'])
        ]
        mpra_df.loc[mpra_triple_target_df.index, 'target'] = get_targets_as_tuple(mpra_triple_target_df)

    return mpra_df
