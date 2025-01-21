# tests/conftest.py

import pytest
import os
from pyfaidx import Fasta

from orthrus.genome import Genome


@pytest.fixture
def small_fasta(tmp_path):
    """
    Create a small FASTA file and index it with pyfaidx.
    Returns the path to the FASTA file.
    """
    # Create a minimal example with two chromosomes.
    fasta_path = tmp_path / "small.fa"
    contents = (
        ">chr1\n"
        "ACGTACGT\n"  # 8 bases for chr1
        ">chr2\n"
        "TTTTGGGG\n"  # 8 bases for chr2
    )
    fasta_path.write_text(contents)

    # The first time we instantiate Fasta(...), pyfaidx automatically
    # creates a .fai index in the same directory.
    Fasta(str(fasta_path))
    return str(fasta_path)


@pytest.fixture
def genome_obj(small_fasta):
    """
    Return a Genome object backed by our small, indexed FASTA.
    """
    return Genome(input_path=small_fasta)
