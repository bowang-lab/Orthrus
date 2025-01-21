# tests/test_interval.py

import pytest
import numpy as np
from orthrus.data import Interval


def test_interval_init_valid(genome_obj):
    interval = Interval("chr1", 2, 6, "+", genome_obj)
    assert interval.chrom == "chr1"
    assert interval.start == 2
    assert interval.end == 6
    assert interval.strand == "+"
    assert len(interval) == 4


def test_interval_repr(genome_obj):
    interval = Interval("chr2", 0, 4, "-", genome_obj)
    r = repr(interval)
    assert "Interval chr2:0-4:-" in r


def test_interval_overlaps(genome_obj):
    i1 = Interval("chr1", 1, 5, "+", genome_obj)
    i2 = Interval("chr1", 3, 7, "+", genome_obj)
    i3 = Interval("chr1", 5, 9, "+", genome_obj)
    i4 = Interval("chr1", 3, 7, "-", genome_obj)

    # Overlap of i1 (1..5) and i2 (3..7) => 2 base overlap (3..5)
    assert i1.overlaps(i2) == 2
    # i1 vs i3 => adjacency at 5 => overlap = 0
    assert i1.overlaps(i3) == 0
    # Different strands => overlap=0
    assert i1.overlaps(i4) == 0


def test_interval_one_hot_encode(genome_obj):
    interval = Interval("chr1", 0, 4, "+", genome_obj)
    oh = interval.one_hot_encode(zero_mean=False)
    assert oh.shape == (4, 4)
    # "ACGT" => rows=4 => check row0 => A => [1,0,0,0]
    assert np.array_equal(oh[0], [1, 0, 0, 0])


def test_interval_sequence(genome_obj):
    interval = Interval("chr2", 2, 6, "-", genome_obj)
    seq = interval.sequence()
    # chr2 => "TTTTGGGG"
    # slice => [2..6] => "TTGG", then reversed & complemented => "CCAA"
    assert seq == "CCAA"
