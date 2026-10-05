from collections import defaultdict
from itertools import product

import numpy
import pandas
import scipy

import Bio
import Bio.Seq
import Bio.SeqRecord
import Bio.SeqIO

import editdistance

def one_hot_encode(sequences, max_seq_len=None, mask_val=0, padding='left', verbose_idx=None):
    """
    One-hot encodes a list of sequences.

    Parameters
    ----------
    sequences : list of str
        List of sequences to one-hot encode.
    max_seq_len : int, optional
        Maximum length of sequences. If not specified, the maximum length of the input sequences will be used.
    mask_val : int, optional
        Value to use for masking. Default is 0.
    padding : str, optional
        Where to pad sequences. Options are 'left', 'right', or 'center'. Default is 'left'.
    verbose_idx : int, optional
        If specified, will print progress every `verbose_idx` sequences. Default is None.

    Returns
    -------
    numpy.ndarray
        One-hot encoded sequences, with shape (n_sequences, max_seq_len, 4).

    """

    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'a':[1,0,0,0],
             'c':[0,1,0,0],
             'g':[0,0,1,0],
             't':[0,0,0,1],
             'n':[0,0,0,0]}

    # Automatically use max length if not specified
    if max_seq_len is None:
        max_seq_len = numpy.max([len(s) for s in sequences])

    # Creat empty matrix
    one_hot_seqs = numpy.ones([len(sequences), max_seq_len, 4])*mask_val
    
    # Iterate through sequences and one-hot encode
    for i, seq in enumerate(sequences):
        if verbose_idx is not None and i%verbose_idx==0:
            print(f'Encoding sequence {i + 1}/{len(sequences)}')
        # Truncate if necessary
        if len(seq)>max_seq_len:
            if padding=='left':
                seq = seq[:max_seq_len]
            elif padding=='right':
                seq = seq[-max_seq_len:]
            elif padding=='center':
                seq = seq[(len(seq)-max_seq_len)//2:(len(seq)+max_seq_len)//2]
            else:
                raise ValueError(f'padding {padding} not recognized')
        # Convert to array
        seq = seq.lower()
        one_hot_seq = numpy.array([nuc_d.get(x, [0, 0, 0, 0]) for x in seq])
        # Append to matrix
        if padding=='left':
            one_hot_seqs[i, :len(seq), :] = one_hot_seq
        elif padding=='right':
            one_hot_seqs[i, -len(seq):, :] = one_hot_seq
        elif padding=='center':
            one_hot_seqs[i, (max_seq_len-len(seq))//2:(max_seq_len+len(seq))//2, :] = one_hot_seq
        else:
            raise ValueError(f'padding {padding} not recognized')
            
    return one_hot_seqs

def revcomp_seq(seq):
    """
    Returns the reverse complement of a sequence.

    Parameters
    ----------
    seq : str
        Sequence to reverse complement.
    
    Returns
    -------
    str
        Reverse complement of the input sequence.

    """
    
    revcomp_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([revcomp_dict[nt] for nt in seq[::-1]])

def one_hot_decode(onehots):
    """
    Converts a set of one-hot encoded sequences to strings.

    Parameters
    ----------
    onehots : 2D numpy.ndarray or list
        One-hot encoded sequences, with shape (seq_len, 4).
    
    Returns
    -------
    str
        String representation of the one-hot encoded sequence.

    Notes
    -----
        This function only handles ACGT.

    """

    seq_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seqs = []
    for onehot in onehots:
        seqs.append(''.join([seq_dict[numpy.argmax(x)] for x in onehot]))

    return seqs

def load_seqs_from_fasta(filepath, id_col='id', seq_col='seq'):
    """
    Retrieves sequences from a fasta file in a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the fasta file.
    id_col : str, optional
        Name of the column to store the sequence id. Default is 'id'.
    seq_col : str, optional
        Name of the column to store the sequence. Default is 'seq'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the sequence ids and sequences.

    """

    # Read from fasta
    seq_ids = []
    seqs = []
    fasta_sequences = Bio.SeqIO.parse(open(filepath),'fasta')
    for fasta in fasta_sequences:
        seqs.append(str(fasta.seq))
        seq_ids.append(fasta.id)
    # Make output dataframe
    seqs_df = pandas.DataFrame(
        {
            id_col: seq_ids,
            seq_col: seqs,
        }
    )
    return seqs_df

def save_seqs_to_fasta(seqs_df, filepath, id_col='id', seq_col='seq'):
    """
    Saves sequences from a pandas DataFrame into a fasta file.

    Parameters
    ----------
    seqs_df : pandas.DataFrame
        DataFrame containing the sequence ids and sequences.
    filepath : str
        Path to save the fasta file.
    id_col : str, optional
        Name of the column containing the sequence id. Default is 'id'.
    seq_col : str, optional
        Name of the column containing the sequence. Default is 'seq'.

    """

    # Extract sequences and ids from DataFrame
    seq_records = []
    for index, row in seqs_df.iterrows():
        seq_records.append(
            Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(row[seq_col]), id=row[id_col], description="")
        )

    # Actually save
    with open(filepath, "w") as output_handle:
        Bio.SeqIO.write(seq_records, output_handle, "fasta")

def get_longest_repeat(s):
    """
    Returns the length of the longest repeat in a sequence.

    Parameters
    ----------
    s : str
        Sequence to analyze.

    Returns
    -------
    int
        Length of the longest repeat in the input sequence.

    """
    repetitions = 0
    while any(s):
        repetitions += 1
        s = [ s[i] and s[i] == s[i+1] for i in range(len(s)-1) ]

    return repetitions

def get_paired_editdistances(seqs, len_norm='max', random_seed=None):
    """
    Returns the edit distances between random pairs of sequences in a list.

    Parameters
    ----------
    seqs : list of str
        List of sequences to compare.
    len_norm : str, optional
        Normalization method for edit distance. The default is 'max'.
    random_seed : int, optional
        Random seed for shuffling sequences. The default is None

    Returns
    -------
    numpy.ndarray
        Array of edit distances, at least one pair per provided sequence.

    """
    if random_seed is not None:
        numpy.random.seed(random_seed)

    shuffle_index = numpy.arange(len(seqs))
    
    # Reject shufflings if any element remains in its original position
    while numpy.any(shuffle_index==numpy.arange(len(seqs))):
        numpy.random.shuffle(shuffle_index)

    distances = []
    for i in range(len(seqs)) :
        if i == shuffle_index[i] :
            continue
        seq_1 = seqs[i]
        seq_2 = seqs[shuffle_index[i]]
        dist = editdistance.eval(seq_1, seq_2)
        # dist = dist / ((len(seq_1) + len(seq_2)) / 2)
        if len_norm == 'max':
            dist = dist / max(len(seq_1), len(seq_2))
        elif len_norm == 'mean':
            dist = dist / ((len(seq_1) + len(seq_2)) / 2)
        elif len_norm == 'min':
            dist = dist / min(len(seq_1), len(seq_2))
        elif len_norm == 'none':
            pass
        else:
            raise ValueError(f"Invalid normalization method: {len_norm}")
        distances.append(dist)

    distances = numpy.array(distances)

    return distances

def _get_all_nmers(n, alphabet='ACGT'):
    """
    Generate all possible n-mers from a given alphabet.

    Parameters
    ----------
    n : int
        Length of n-mer.
    alphabet : str, optional
        Alphabet to use for generating n-mers. Default is 'ACGT'.

    Returns
    -------
    list
        List of all possible n-mers of length n.

    """
    return [''.join(p) for p in product(alphabet, repeat=n)]

def _str2nmer(str, nmer):
    """
    Convert string to a list of n-mers

    Parameters
    ----------
    str : str
        Input string to convert.
    nmer : int
        Length of n-mer.

    Returns
    -------
    list
        List of n-mers extracted from the input string, in the order they occur.

    """
    return [str[i:i+nmer] for i in range(len(str)-nmer+1)]

def _count_nmers(nmers_list, nmer_keys):
    """
    Count occurrences of each n-mer in the input list of nmers.

    Parameters
    ----------
    nmers_list : list of str
        List of nmers representing a single string
    nmer_keys : list of str
        List of all possible nmers

    Returns
    -------
    nmers_vector: list
        List of counts for each n-mer in the input list, in the order of nmer_keys.

    """

    # Initialize a dictionary to store the counts - default fills in 0 for missing keys
    counts = defaultdict(int)
    
    # populate the dictionary with counts
    for s in nmers_list:
        counts[s] += 1
    
    # Create a vector of counts for all 4-mers
    nmers_vector = [counts[mer] for mer in nmer_keys]

    return nmers_vector

def _get_min_euc_dists(vecsA, vecsB, subsampleB=None, random_seed_subsample=None):
    """
    Calculate the minimum Euclidean distances between two sets of nmer vectors.

    For each sequence in vecsA, this function calculates the minimum Euclidean
    distance to sequences from vecsB. As a default, all sequences in vecsB are used,
    but a random subsample can be specified.

    Parameters
    ----------
    vecsA : list of numpy.ndarray
        First set of nmer vectors.
    vecsB : list of numpy.ndarray
        Second set of nmer vectors.
    subsampleB : int, optional
        Number of sequences to randomly sample from vecsB. Default is None.
    random_seed_subsample : int, optional
        Random seed for subsampling vecsB. Default is None.

    Returns
    -------
    numpy.ndarray
        Array of minimum distances for each sequence in vecsA.

    """
    # distances will be calculated with respect to vecsA
    if subsampleB is not None:
        if subsampleB > len(vecsB):
            raise ValueError("Subsample size is greater than the number of sequences in vecsB.")
        numpy.random.seed(random_seed_subsample)
        vecsB = vecsB[numpy.random.choice(len(vecsB), subsampleB, replace=False)]

    n_seqsA = len(vecsA)
    n_seqsB = len(vecsB)
    min_distances = numpy.zeros(n_seqsA)
    
    for i in range(n_seqsA):
        
        distances = numpy.zeros(n_seqsB)
        for j in range(n_seqsB):
            distances[j] = scipy.spatial.distance.euclidean(vecsA[i], vecsB[j])
            
        # don't need to normalize here
        min_distances[i] = numpy.min(distances[distances != 0])
        
    return min_distances

def get_min_euc_nmer_dist(seqsA, seqsB=None, nmer=4, subsampleB=None, random_seed_subsample=None, normalize_counts=False):
    """
    Calculate the minimum Euclidean distances between two sets of sequences.

    For each sequence in seqsA, this function calculates the minimum Euclidean
    distance to sequences from seqsB. As a default, all sequences in seqsB are used,
    but a random subsample can be specified.

    Parameters
    ----------
    seqsA : list of str
        First set of sequences.
    seqsB : list of str, optional
        Second set of sequences. If None, distance will be calcualted with respect to seqsA.
    nmer : int, optional
        Length of n-mer. Default is 4.
    subsampleB : int, optional
        Number of sequences to randomly sample from seqsB. Default is None.
    random_seed_subsample : int, optional
        Random seed for subsampling seqsB. Default is None.

    Returns
    -------
    numpy.ndarray
        Array of minimum distances for each sequence in seqsA.

    """
    
    # Generate all possible n-mers
    nmer_keys = _get_all_nmers(nmer)
    
    # Count n-mers in both sets of sequences
    seqsA_nmer_counts = numpy.array([_count_nmers(_str2nmer(s, nmer), nmer_keys) for s in seqsA])
    if normalize_counts:
        seqsA_nmer_counts = seqsA_nmer_counts / numpy.sum(seqsA_nmer_counts, axis=1, keepdims=True)
    
    if seqsB is not None:
        seqsB_nmer_counts = numpy.array([_count_nmers(_str2nmer(s, nmer), nmer_keys) for s in seqsB])
        if normalize_counts:
            seqsB_nmer_counts = seqsB_nmer_counts / numpy.sum(seqsB_nmer_counts, axis=1, keepdims=True)
    else:
        seqsB_nmer_counts = seqsA_nmer_counts
    
    return _get_min_euc_dists(
        seqsA_nmer_counts,
        seqsB_nmer_counts,
        subsampleB=subsampleB,
        random_seed_subsample=random_seed_subsample,
    )
