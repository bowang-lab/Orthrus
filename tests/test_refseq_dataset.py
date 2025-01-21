# tests/test_refseq_dataset.py

import pytest
import numpy as np
import pandas as pd
from orthrus.data import Transcript, RefseqDataset


def test_refseq_dataset_init(genome_obj):
    t1 = Transcript("NM_01", "G1", [0], [4], genome_obj, 1, "+", "chr1", 0, 4, 0, 4)
    t2 = Transcript("NM_02", "G2", [0], [4], genome_obj, 1, "-", "chr2", 0, 4, 0, 4)
    ds = RefseqDataset([t1, t2])
    assert len(ds) == 2
    # The max length among the two transcripts => 4
    assert ds.max_transcript_length == 4


def test_refseq_dataset_one_hot_encode_dataset(genome_obj):
    t1 = Transcript("NM_01", "G1", [0], [4], genome_obj, 1, "+", "chr1", 0, 4, 0, 4)
    t2 = Transcript("NM_02", "G2", [0], [4], genome_obj, 1, "-", "chr2", 0, 4, 0, 4)
    ds = RefseqDataset([t1, t2])
    oh_dataset = ds.one_hot_encode_dataset(pad_length_to=6, zero_mean=False)
    # => shape => (2,6,4)
    assert oh_dataset.shape == (2, 6, 4)
