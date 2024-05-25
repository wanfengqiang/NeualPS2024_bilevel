import torch
from torch.utils.data import Dataset, DataLoader
import random
import time
from datetime import datetime
from functools import partial
import json
from tqdm import tqdm
import numpy as np

def basic_sampler(seq, sample_len):
    """
    Basic text sampler.
    Returns the first sample_len items.
    If sample_len is greater than the length of the seq, the seq is returned.
    """
    seq_len = len(seq)
    if seq_len > sample_len:
        return seq[:sample_len]
    else:
        return seq
    

def basic_rand_sampler(seq, sample_len):
    """
    Basic random text sampler.
    If sample_len is greater than the length of the seq, the seq is returned.
    """
    seq_len   = len(seq)
    if seq_len > sample_len:
        start_idx = random.randint(0, min(seq_len,seq_len - sample_len))
        end_idx   = start_idx+sample_len
        return seq[start_idx:end_idx]
    else:
        return seq
    

