# tests/test_transcript.py

import pytest
import numpy as np
from orthrus.data import Transcript


def test_transcript_init(genome_obj):
    # Let's define exons on chr1: 0..4 and 4..8 => total 8bp
    t = Transcript(
        transcript_id="NM_0001",
        gene="TEST",
        exon_starts=[0, 4],
        exon_ends=[4, 8],
        genome=genome_obj,
        exon_count=2,
        strand="+",
        chromosome="chr1",
        tx_start=0,
        tx_end=8,
        cds_start=0,
        cds_end=8,
    )
    assert t.chromosome == "chr1"
    assert len(t) == 8  # sum of exons => 4 + 4 = 8


def test_transcript_one_hot_encode_transcript(genome_obj):
    # Similar setup
    t = Transcript(
        "NM_0002",
        "TEST2",
        exon_starts=[0, 4],
        exon_ends=[4, 8],
        genome=genome_obj,
        exon_count=2,
        strand="+",
        chromosome="chr1",
        tx_start=0,
        tx_end=8,
        cds_start=0,
        cds_end=8,
    )
    oh = t.one_hot_encode_transcript(pad_length_to=10, zero_mean=False, zero_pad=False)
    assert oh.shape == (10, 4)
    # The last 2 rows are padding => [0.25,0.25,0.25,0.25]


def test_transcript_get_amino_acid_sequence(genome_obj):
    # If cds_start=0..cds_end=6 => 2 codons => 'ACGTAC' => let's see
    # Over-simplification for demonstration:
    t = Transcript(
        transcript_id="NM_0003",
        gene="GENE3",
        exon_starts=[0],
        exon_ends=[6],
        genome=genome_obj,
        exon_count=1,
        strand="+",
        chromosome="chr1",
        tx_start=0,
        tx_end=6,
        cds_start=0,
        cds_end=6,
    )
    # The genomic sequence for chr1 => "ACGTACGT" => but we only get indices 0..6 => "ACGTAC"
    # "ACG" => codon => T1?
    # "TAC" => codon => Y?
    # Let's see how your `translate_dna()` method handles "ACGTAC"
    aa_seq = t.get_amino_acid_sequence()
    # Just check it's not empty, and is length 2 if no stop codons appear
    assert len(aa_seq) == 2
