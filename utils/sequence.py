import numpy

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
