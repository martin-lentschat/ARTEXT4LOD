# coding: utf-8
from __future__ import unicode_literals

import warnings

import numpy as np


def match_sequence(seq, data_set):
    """
    Return match sequence start,end positions in a dataset
    Parameters
    ----------
    seq : list
        sequence
    data_set : list
        dataset

    """
    n = len(seq)
    if n < 1:
        raise ValueError("Sequence is empty !")

    if isinstance(data_set, list):
        data_set = np.asarray(data_set)
    if isinstance(seq, list):
        seq = np.asarray(seq)
    prefix_ind = np.where(data_set == seq[0])[0]
    results = []
    for idx in prefix_ind:
        start, end = idx, idx + n
        if data_set[start:end].tolist() == seq.tolist():
            results.append([data_set[start:end], start, end])
    return results


def match_sequences(seqs, data_set):
    """
    Return match sequence start,end positions in a dataset

    Parameters
    ----------
    seqs : list
        sequence
    data_set : list
    """
    n = len(seqs)
    if n < 1:
        warnings.warn("Sequence Empty")
        return []

    if isinstance(data_set, list):
        data_set = np.asarray(data_set)
    if isinstance(seqs, list):
        seqs = np.array(seqs)

    prefixes_dict = {seq[0]: (seq, i, len(seq)) for i, seq in enumerate(seqs)}
    prefixes = list(prefixes_dict.keys())
    prefix_ind = np.where(np.isin(data_set, prefixes))[0]
    results = []
    for idx in prefix_ind:
        start, end = idx, idx + prefixes_dict[data_set[idx]][-1]
        if data_set[start:end].tolist() == prefixes_dict[data_set[idx]][0]:
            results.append([prefixes_dict[data_set[idx]][1], start, end])
    return results


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


if __name__ == "__main__":
    print('matcher')
