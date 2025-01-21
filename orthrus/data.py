import os
from pathlib import Path

__all__ = ["Interval", "Transcript", "RefseqDataset"]

import pandas as pd
import numpy as np
from .genome import Genome
from typing import List


class Interval(object):
    """
    Interval for representing genomic coordinates.
    Uses 0-start, half-open (0-based) coordinate system. Same as ucsc genome browser internal documentation
    For more information:
        - http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/
        - https://www.biostars.org/p/84686/

    """

    def __init__(
        self, chromosome: str, start: int, end: int, strand: str, genome: Genome
    ):
        """
        Initialize an Interval object.

        Args:
            chromosome (str): Chromosome name (e.g., 'chr1', 'chr2', etc.).
            start (int): 0-based start coordinate (inclusive).
            end (int): 0-based end coordinate (exclusive).
            strand (str): Strand of the interval ('+' or '-').
            genome (Genome): Genome object providing sequence/encoding methods.
        """
        assert strand in ["+", "-"], strand
        self.strand = strand
        self.chrom = chromosome

        assert start <= end, (start, end)
        assert start >= 0, (start, end)
        self.start = start
        self.end = end

        self.genome = genome
        self.alphabet_map = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}

    def __len__(self):
        """
        Get the length of the interval.

        Returns:
            int: The length of the interval (end - start).
        """
        return self.end - self.start

    @property
    def chromosome(self):
        """
        Chromosome property for easier usage in string formatting.

        Returns:
            str: The chromosome name of this interval.
        """
        return self.chrom

    def __repr__(self):
        """
        Return a string representation of the Interval.

        Returns:
            str: String representation of the interval in the form:
                 'Interval chrom:start-end:strand'.
        """
        return "Interval {}:{}-{}:{}".format(
            self.chromosome, self.start, self.end, self.strand
        )

    def overlaps(self, interval):
        """
        Compute the overlap in base pairs between this interval and another interval.

        Args:
            interval (Interval): Another interval to check overlap with.

        Returns:
            int: Number of overlapping bases. Returns 0 if there is no overlap,
                 or if they are on different chromosomes/strands.
        """
        assert isinstance(interval, Interval)
        if interval.chrom != self.chrom:
            return 0
        if interval.strand != self.strand:
            return 0

        overlap_start = np.max([self.start, interval.start])
        overlap_end = np.min([self.end, interval.end])

        overlap = overlap_end - overlap_start
        if overlap < 0:
            return 0
        else:
            return overlap

    def within(self, interval):
        """
        Check whether this interval is fully contained within another interval.

        Args:
            interval (Interval): The interval in which to check containment.

        Returns:
            bool: True if this interval is within the given interval, False otherwise.
        """
        assert isinstance(interval, Interval)
        if interval.chrom != self.chrom:
            return False
        if interval.strand != self.strand:
            return False

        after_start = self.start >= interval.start
        before_end = self.end <= interval.end

        return after_start and before_end

    def one_hot_encode(self, zero_mean: bool = True):
        """
        Get one-hot encoding of the sequence within this interval.

        Args:
            zero_mean (bool, optional): Whether to apply zero-mean normalization
                by subtracting 0.25 from each channel. Defaults to True.

        Returns:
            np.ndarray: A NumPy array of shape (length_of_interval, 4) containing
                one-hot or zero-mean-encoded representation of the sequence.
        """
        seq = self.genome.get_encoding_from_coords(
            self.chrom, self.start, self.end, self.strand
        )
        if zero_mean:
            seq = seq - 0.25
        return seq

    def sequence(self):
        """
        Get the nucleotide sequence for this interval as a string.

        Returns:
            str: The nucleotide sequence corresponding to this interval.
        """
        return self.genome.get_sequence_from_coords(
            self.chrom, self.start, self.end, self.strand
        )

    def encode(self):
        """
        Encode the nucleotide sequence of this interval into numeric labels.

        Returns:
            np.ndarray: A NumPy array of shape (length_of_interval,) containing
                integer labels (0 for A, 1 for C, 2 for T, 3 for G, 4 for N).
        """
        seq = self.genome.get_sequence_from_coords(
            self.chrom, self.start, self.end, self.strand
        ).upper()
        return np.array([self.alphabet_map[x] for x in seq])


class Transcript(object):
    """
    An object representing an RNA transcript, allowing queries for its sequence,
    conversions to one-hot encoding, and various transcript-level utilities.
    """

    def __init__(
        self,
        transcript_id: str,
        gene: str,
        exon_starts: List[int],
        exon_ends: List[int],
        genome: Genome,
        exon_count: int,
        strand: str,
        chromosome: str,
        tx_start: int,
        tx_end: int,
        cds_start: int,
        cds_end: int,
        expand_transcript_distance: int = 0,
        expand_exon_distance: int = 0,
    ):
        """
        Initialize a Transcript object.

        Args:
            transcript_id (str): Identifier for the transcript (e.g., 'NM_001...').
            gene (str): Gene name associated with the transcript.
            exon_starts (List[int]): List of start coordinates for each exon.
            exon_ends (List[int]): List of end coordinates for each exon.
            genome (Genome): Genome object providing sequence/encoding methods.
            exon_count (int): Number of exons in this transcript.
            strand (str): Strand of the transcript ('+' or '-').
            chromosome (str): Chromosome on which the transcript resides.
            tx_start (int): Start coordinate of the transcript (inclusive).
            tx_end (int): End coordinate of the transcript (exclusive).
            cds_start (int): Start coordinate of the coding sequence (inclusive).
            cds_end (int): End coordinate of the coding sequence (exclusive).
            expand_transcript_distance (int, optional): Distance to extend
                upstream/downstream of the transcript. Defaults to 0.
            expand_exon_distance (int, optional): Distance to extend
                around each exon. Defaults to 0.
        """
        self.transcript_id = str(transcript_id)
        self.gene = str(gene)
        self.chromosome = chromosome
        self.chrom = chromosome
        self.strand = strand
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.exon_starts = exon_starts
        self.exon_ends = exon_ends
        self.exon_count = exon_count

        self.expand_transcript_distance = expand_transcript_distance
        self.expand_exon_distance = expand_exon_distance
        self.genome = genome

        assert self.strand in ["+", "-"]
        assert len(self.exon_ends) == len(self.exon_starts), (
            len(self.exon_ends),
            len(self.exon_starts),
            transcript_id,
        )
        assert len(self.exon_ends) == exon_count, (
            len(self.exon_ends),
            exon_count,
            transcript_id,
        )

        # Construct exon expand distance
        self.exon_distance_list = self.generate_inter_exon_distances()
        self.exon_expand_distances = self.calculate_expand_distance()

        # construct transcript intervals which includes the exon expand
        # distance and transcript expand distance
        self.transcript_intervals = self.construct_transcript_intervals()

        # Construct exon list
        exons = []
        for exon_coords in zip(exon_starts, exon_ends):
            exons.append(
                Interval(
                    self.chromosome,
                    exon_coords[0],
                    exon_coords[1],
                    self.strand,
                    self.genome,
                )
            )
        self.exons = exons

    def __len__(self):
        """
        Return the total length of all intervals that make up this transcript,
        including any exon/transcript expansions.

        Returns:
            int: The sum of lengths of each interval in transcript_intervals.
        """
        return np.sum([len(x) for x in self.transcript_intervals])

    def __repr__(self) -> str:
        """
        Return a string representation of this transcript.

        Returns:
            str: String of the form 'Transcript id chrom:tx_start-tx_end:strand'.
        """
        return "Transcript {} {}:{}-{}:{}".format(
            self.transcript_id, self.chromosome, self.tx_start, self.tx_end, self.strand
        )

    def __hash__(self):
        """
        Create a hash of this transcript object for usage in sets/dicts.

        Returns:
            int: A hash representing unique aspects of the transcript.
        """
        identifier = (
            f"{self.gene} length {self.__len__()} n_exons {self.exon_count}"
            f"{self.transcript_id} {self.chromosome}:{self.tx_start}-{self.tx_end}:{self.strand}"
        )
        return hash(identifier)

    def __eq__(self, other):
        """
        Check equality with another Transcript object.

        Args:
            other (Transcript): Another transcript to compare with.

        Returns:
            bool: True if all transcript attributes match, False otherwise.
        """

        gene_eq = self.gene == other.gene
        len_eq = self.__len__() == other.__len__()
        exn_cnt_eq = self.exon_count == other.exon_count
        trsc_id_eq = self.transcript_id == other.transcript_id
        chrom_eq = self.chromosome == other.chromosome
        tx_start_eq = self.tx_start == other.tx_start
        tx_end_eq = self.tx_end == other.tx_end
        stand_eq = self.strand == other.strand
        conditions = [
            gene_eq,
            len_eq,
            exn_cnt_eq,
            trsc_id_eq,
            chrom_eq,
            tx_start_eq,
            tx_end_eq,
            stand_eq,
        ]
        return all(conditions)

    def generate_inter_exon_distances(self):
        """
        Generate the distances between each exon based on exon_starts/exon_ends.

        Returns:
            list of tuples: Each tuple contains the distance to the next exon
                and the distance to the previous exon, for each exon in the transcript.
        """
        inter_exon_dist_tuple = list(
            zip([x for x in self.exon_starts[:-1]], [x for x in self.exon_ends[1:]])
        )
        inter_exon_dist = [x[1] - x[0] for x in inter_exon_dist_tuple]

        distances_to_next_exon = inter_exon_dist + [(self.tx_end - self.exon_ends[-1])]
        distances_to_previous_exon = [
            self.exon_starts[0] - self.tx_start
        ] + inter_exon_dist
        exon_distance_list = list(
            zip(distances_to_next_exon, distances_to_previous_exon)
        )

        # Require all the elements to not be intersecting
        assert all([dist >= 0 for x in exon_distance_list for dist in x]), (
            self.transcript_id,
            self.chromosome,
            self.tx_start,
            self.tx_end,
            self.strand,
            self.exon_starts,
            self.exon_ends,
            exon_distance_list,
        )
        return exon_distance_list

    def calculate_relative_cds_start_end(self):
        """
        Calculate the coding sequence (CDS) start and end relative to the
        concatenated transcript_intervals.

        Returns:
            tuple: (relative_cds_start, relative_cds_end),
                   the 0-based indices within the full transcript for CDS start/end.

        Raises:
            AssertionError: If expand_exon_distance or expand_transcript_distance is non-zero,
                            indicating the method is not valid for transcripts with introns/promoters expanded.
        """
        assert self.expand_exon_distance == 0, "Doesn't work with introns"
        assert self.expand_transcript_distance == 0, "Doesn't work with promoters"

        first_cds_base = Interval(
            self.chromosome, self.cds_start, self.cds_start, self.strand, self.genome
        )
        last_cds_base = Interval(
            self.chromosome, self.cds_end, self.cds_end, self.strand, self.genome
        )

        assert self.transcript_intervals

        # Find CDS start
        exon_index_cds_start = np.argwhere(
            [first_cds_base.within(interval) for interval in self.transcript_intervals]
        )
        assert len(exon_index_cds_start) == 1
        exon_index_cds_start = exon_index_cds_start[0][0]

        # count all the exons until the start
        length_until_cds_start = sum(
            [len(x) for x in self.transcript_intervals[: exon_index_cds_start + 1]]
        )
        # subtract distance from end of exon until start of CDS
        within_exon_d_to_start = (
            self.transcript_intervals[exon_index_cds_start].end - first_cds_base.end
        )
        assert within_exon_d_to_start >= 0
        length_until_cds_start -= within_exon_d_to_start

        # Find CDS end
        exon_index_cds_end = np.argwhere(
            [last_cds_base.within(interval) for interval in self.transcript_intervals]
        )
        assert len(exon_index_cds_end) == 1
        exon_index_cds_end = exon_index_cds_end[0][0]

        length_until_cds_end = sum(
            [len(x) for x in self.transcript_intervals[: exon_index_cds_end + 1]]
        )
        # subtract distance from end of exon until start of CDS
        within_exon_d_to_end = (
            self.transcript_intervals[exon_index_cds_end].end - last_cds_base.end
        )
        assert within_exon_d_to_end >= 0
        length_until_cds_end -= within_exon_d_to_end

        if self.strand == "+":
            relative_cds_end = length_until_cds_end
            relative_cds_start = length_until_cds_start
        elif self.strand == "-":
            relative_cds_end = self.__len__() - length_until_cds_start
            relative_cds_start = self.__len__() - length_until_cds_end

        return relative_cds_start, relative_cds_end

    def calculate_relative_splice_sites(self):
        """
        Calculate the relative indices of the splice sites within the
        concatenated transcript_intervals.

        Returns:
            list: A list of 0-based indices representing the last nucleotide
                  in each exon within the full transcript.

        Raises:
            AssertionError: If expand_exon_distance or expand_transcript_distance is non-zero,
                            meaning splice sites might not align with expansions.
        """
        assert self.expand_exon_distance == 0, "Doesn't work with introns"
        assert self.expand_transcript_distance == 0, "Doesn't work with promoters"
        assert self.transcript_intervals

        if self.strand == "-":

            indices = [
                sum([len(x) for x in self.transcript_intervals[:exon_index]])
                for exon_index in range(len(self.transcript_intervals))
            ]
            indices = [len(self) - x for x in indices]
            indices = indices[::-1]
            # 0 base encoding
            indices = [x - 1 for x in indices]

        elif self.strand == "+":
            indices = [
                sum([len(x) for x in self.transcript_intervals[: exon_index + 1]])
                for exon_index in range(len(self.transcript_intervals))
            ]
            # 0 base encoding
            indices = [x - 1 for x in indices]

        return indices

    def calculate_expand_distance(self):
        """
        Calculate expansion distance around each exon, ensuring there are no
        overlaps between adjacent expanded regions.

        Returns:
            list of tuples: Each tuple (up_expansion, down_expansion) gives the
                expansion distance for each exon.
        """
        exon_expand_distances = []
        for i in range(self.exon_count):
            # 2 cases
            #   1 where the distance between exons is greater than 2 * exon_expand_distance
            #   2 where the distance between exons is less than 2 * exon_expand_distance

            # in case of first exon the tx start won't be expanding forwards so we don't have to impose /2 restriction
            if i != 0:
                free_distance_to_previous_exon = self.exon_distance_list[i][0] / 2
            else:
                free_distance_to_previous_exon = self.exon_distance_list[i][0]

            if free_distance_to_previous_exon < self.expand_exon_distance:
                # Take half of the available distance to the next exon
                expand_exon_distance_up = int(free_distance_to_previous_exon)
            else:
                expand_exon_distance_up = self.expand_exon_distance

            # in case of last exon the tx_end won't be expanding backwards so we don't have to impose /2 restriction
            if i != self.exon_count - 1:
                free_distance_to_next_exon = self.exon_distance_list[i][1] / 2
            else:
                free_distance_to_next_exon = self.exon_distance_list[i][1]

            if free_distance_to_next_exon < self.expand_exon_distance:
                # In case the next exon is closer than the expand distance take half the available space to expand
                expand_exon_distance_down = int(free_distance_to_next_exon)
            else:
                expand_exon_distance_down = self.expand_exon_distance

            exon_expand_distances.append(
                (expand_exon_distance_up, expand_exon_distance_down)
            )
        return exon_expand_distances

    def construct_transcript_intervals(self):
        """
        Construct Interval objects spanning the entire transcript,
        including exon expansions and transcript expansions.

        Returns:
            list: A list of Interval objects covering the transcript and expansions.
        """
        transcript_intervals = []

        # transcript expand distance
        if self.expand_transcript_distance:
            transcript_intervals.append(
                Interval(
                    self.chromosome,
                    self.tx_start - self.expand_transcript_distance,
                    self.tx_start,
                    self.strand,
                    self.genome,
                )
            )

        # Exon intervals
        for i in range(self.exon_count):
            transcript_intervals.append(
                Interval(
                    self.chromosome,
                    self.exon_starts[i] - self.exon_expand_distances[i][0],
                    self.exon_ends[i] + self.exon_expand_distances[i][1],
                    self.strand,
                    self.genome,
                )
            )

        if self.expand_transcript_distance:
            transcript_intervals.append(
                Interval(
                    self.chromosome,
                    self.tx_end,
                    self.tx_end + self.expand_transcript_distance,
                    self.strand,
                    self.genome,
                )
            )

        return transcript_intervals

    def one_hot_encode_transcript(
        self, pad_length_to: int = 0, zero_mean: bool = False, zero_pad: bool = False
    ):
        """
        Create a one-hot-encoded (or zero-mean) matrix for the entire transcript.

        Args:
            pad_length_to (int, optional): Length to which to pad the sequence.
                If 0, no padding is applied. Defaults to 0.
            zero_mean (bool, optional): Whether to subtract 0.25 from each one-hot
                channel. Defaults to False.
            zero_pad (bool, optional): If True, pads with zeros. If False, pads
                with 0.25. Defaults to False.

        Returns:
            np.ndarray: Concatenated one-hot-encoded array of shape
                (total_transcript_length, 4) or (pad_length_to, 4).
        """
        if self.strand == "+":
            one_hot_list = [
                x.one_hot_encode(zero_mean) for x in self.transcript_intervals
            ]
        elif self.strand == "-":
            one_hot_list = [
                x.one_hot_encode(zero_mean) for x in self.transcript_intervals[::-1]
            ]
        else:
            raise ValueError

        if pad_length_to:
            # check padding length is greater than the self length
            assert (
                len(self) <= pad_length_to
            ), f"Length of transcript {len(self)} greater than padding specified {pad_length_to}"
            # N is represented as [0.25, 0.25, 0.25, 0.25]

            if len(self) < pad_length_to:
                pad_sequence = np.zeros(
                    (pad_length_to - len(self), 4), dtype=np.float32
                )
                # can be not zero mean but still pad with zeros
                if not zero_mean and not zero_pad:
                    pad_sequence = pad_sequence + 0.25

                one_hot_list.append(pad_sequence)

        concat_sequence = np.concatenate(one_hot_list)

        return concat_sequence

    def get_sequence(self, pad_length_to: int = 0):
        """
        Retrieve the concatenated nucleotide sequence for this transcript.

        Args:
            pad_length_to (int, optional): If > 0, pads the sequence to this length
                by adding 'N' characters. Defaults to 0.

        Returns:
            str: The nucleotide sequence. If padding is applied, the padded portion
                is represented by 'N's.
        """

        if self.strand == "+":
            seqs = [x.sequence() for x in self.transcript_intervals]
        elif self.strand == "-":
            seqs = [x.sequence() for x in self.transcript_intervals[::-1]]

        if pad_length_to:
            # check padding length is greater than the self length
            assert (
                len(self) <= pad_length_to
            ), "Length of transcript {} greater than padding specified {}".format(
                len(self), pad_length_to
            )
            # N is represented as [0.25, 0.25, 0.25, 0.25]
            if len(self) < pad_length_to:
                seqs.append((pad_length_to - len(self)) * "N")
        return "".join(seqs)

    @classmethod
    def translate_dna(cls, dna_sequence):
        """
        Translates a DNA sequence into a protein sequence
        using the standard genetic code.

        Args:
            dna_sequence (str): A string representing the
            DNA sequence. Must consist of characters 'T', 'C', 'A', 'G'.

        Returns:
            str: The corresponding protein sequence. Unrecognized
            codons are represented by '?'. The stop codons are represented by '*'.

        Examples:
            >>> translate_dna('ATGGCCATGGCGCCCAGAACTGAGATCAATAGTACCCGTATTAACGGGTGA')
            'MAMAPRTEINSTRING-'
            >>> translate_dna('ATGTTTCAA')
            'MFQ'
        """

        codon_map = {
            "TTT": "F",
            "CTT": "L",
            "ATT": "I",
            "GTT": "V",
            "TTC": "F",
            "CTC": "L",
            "ATC": "I",
            "GTC": "V",
            "TTA": "L",
            "CTA": "L",
            "ATA": "I",
            "GTA": "V",
            "TTG": "L",
            "CTG": "L",
            "ATG": "M",
            "GTG": "V",
            "TCT": "S",
            "CCT": "P",
            "ACT": "T",
            "GCT": "A",
            "TCC": "S",
            "CCC": "P",
            "ACC": "T",
            "GCC": "A",
            "TCA": "S",
            "CCA": "P",
            "ACA": "T",
            "GCA": "A",
            "TCG": "S",
            "CCG": "P",
            "ACG": "T",
            "GCG": "A",
            "TAT": "Y",
            "CAT": "H",
            "AAT": "N",
            "GAT": "D",
            "TAC": "Y",
            "CAC": "H",
            "AAC": "N",
            "GAC": "D",
            "TAA": "*",
            "CAA": "Q",
            "AAA": "K",
            "GAA": "E",
            "TAG": "*",
            "CAG": "Q",
            "AAG": "K",
            "GAG": "E",
            "TGT": "C",
            "CGT": "R",
            "AGT": "S",
            "GGT": "G",
            "TGC": "C",
            "CGC": "R",
            "AGC": "S",
            "GGC": "G",
            "TGA": "*",
            "CGA": "R",
            "AGA": "R",
            "GGA": "G",
            "TGG": "W",
            "CGG": "R",
            "AGG": "R",
            "GGG": "G",
        }

        protein_sequence = ""
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i : i + 3].upper()
            protein_sequence += codon_map.get(codon, "?")

        return protein_sequence

    def get_amino_acid_sequence(self):
        """
        Retrieve the amino acid sequence by translating the coding sequence
        within the transcript.

        Returns:
            str: The translated protein sequence (including '*' for stops).

        Raises:
            AssertionError: If expand_exon_distance or expand_transcript_distance is non-zero,
                            since CDS coordinates assume no expansions.
        """
        rel_cds_start, rel_cds_end = self.calculate_relative_cds_start_end()
        nt_sequence = self.get_sequence()
        coding_sequence = nt_sequence[rel_cds_start:rel_cds_end]
        aa_seq = Transcript.translate_dna(coding_sequence)
        return aa_seq

    def encode(self, pad_length_to: int = 0):
        """
        Encode the transcript nucleotides into numeric labels, optionally padding.

        Args:
            pad_length_to (int, optional): If > 0, pad the encoded array to this length
                with 'N' (represented as 4). Defaults to 0.

        Returns:
            np.ndarray: Numeric array of shape (length_of_transcript,) or (pad_length_to,),
                where {0,1,2,3,4} represent {A,C,T,G,N}.
        """
        if self.strand == "+":
            seqs = np.concatenate(
                [x.encode() for x in self.transcript_intervals]
            ).flatten()
        elif self.strand == "-":
            seqs = np.concatenate(
                [x.encode() for x in self.transcript_intervals[::-1]]
            ).flatten()
        else:
            raise ValueError

        if pad_length_to:
            # check padding length is greater than the self length
            assert (
                len(self) <= pad_length_to
            ), "Length of transcript {} greater than padding specified {}".format(
                len(self), pad_length_to
            )
            # N is represented as [0.25, 0.25, 0.25, 0.25]
            if len(self) < pad_length_to:
                n = np.repeat(4, pad_length_to - len(self))
                seqs = np.concatenate([seqs, n])
        return seqs

    def encode_splice_track(self, pad_length_to: int = 0):
        """
        Encode splice sites as a binary track: 1 at splice site, 0 elsewhere.

        Args:
            pad_length_to (int, optional): If > 0, pad the array to this length.
                Defaults to 0.

        Returns:
            np.ndarray: An array of shape (transcript_length or pad_length_to, 1)
                where splice sites are marked with 1.
        """
        rel_ss = self.calculate_relative_splice_sites()

        if pad_length_to == 0:
            encoding_length = len(self)
        else:
            encoding_length = pad_length_to

        ss_encoded = np.zeros(encoding_length)
        ss_encoded[rel_ss] = 1

        return ss_encoded.reshape(-1, 1)

    def encode_coding_sequence_track(self, pad_length_to: int = 0):
        """
        Encode coding sequence (CDS) positions. Mark the first base of each codon with 1.

        Args:
            pad_length_to (int, optional): Length to which to pad the track.
                Defaults to 0.

        Returns:
            np.ndarray: An array of shape (transcript_length or pad_length_to, 1)
                where the first nucleotide of each codon is marked with 1.
                If there is no CDS, returns all zeros.
        """
        rel_cds_start, rel_cds_end = self.calculate_relative_cds_start_end()

        if pad_length_to == 0:
            encoding_length = len(self)
        else:
            encoding_length = pad_length_to

        # if no coding sequence return empty track
        if rel_cds_start == rel_cds_end:
            return np.zeros((encoding_length, 1))

        first_nuc_index = np.arange(rel_cds_end - rel_cds_start, step=3) + rel_cds_start
        cds_encoded = np.zeros(encoding_length)
        cds_encoded[first_nuc_index] = 1

        return cds_encoded.reshape(-1, 1)

    def encode_6_track(
        self,
        pad_length_to: int = 0,
        zero_mean: bool = False,
        zero_pad: bool = False,
        channels_last: bool = False,
    ):
        """
        Create a 6-channel track consisting of:
            - One-hot-encoded or zero-mean-encoded sequence (4 channels).
            - Coding sequence track (1 channel).
            - Splice site track (1 channel).

        Args:
            pad_length_to (int, optional): If > 0, pad the sequence to this length.
                Defaults to 0.
            zero_mean (bool, optional): If True, subtract 0.25 from each one-hot channel.
                Defaults to False.
            zero_pad (bool, optional): If True, use zeros for padding; if False,
                use 0.25. Defaults to False.
            channels_last (bool, optional): If True, channels are last dimension (L, 6).
                Otherwise, channels are first dimension (6, L). Defaults to False.

        Returns:
            np.ndarray: 6-channel track of shape (6, length) or (length, 6),
                depending on channels_last.
        """
        oh = self.one_hot_encode_transcript(
            pad_length_to=pad_length_to,
            zero_mean=zero_mean,
            zero_pad=zero_pad,
        )
        ss_seq = self.encode_splice_track(pad_length_to=pad_length_to)
        coding_seq = self.encode_coding_sequence_track(pad_length_to=pad_length_to)
        six_track = np.concatenate([oh, coding_seq, ss_seq], axis=1)
        if not channels_last:
            six_track = np.swapaxes(six_track, 0, 1)

        return six_track

    def save_6_track_encoding_npz(
        self,
        directory: str,
        skip_existing: bool = True,
        channels_last: bool = False,
    ):
        """Store compressed transcript six-track encoding.

        Serializes the output of the encode_6_track method using
        np.savez_compressed to a specific directory.

        Args:
            directory: The base directory to save the serialized file.
            channels_last: Stores the channels in last dimension.
            skip_existing: Skips saving if file already exists.
        """
        # Ensure the base directory exists
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        save_name = f"{self.gene.replace('/', '_')}_{self.transcript_id.replace('/', '_')}_channels_last_{channels_last}.npz"
        file_path = dir_path / save_name

        # Generate the file path based on gene and transcript_id
        file_path = dir_path / save_name

        if os.path.exists(file_path) and skip_existing:
            return

        # Encode the 6-track data
        six_track_data = self.encode_6_track(
            pad_length_to=0, zero_mean=False, zero_pad=True, channels_last=False
        ).astype(np.uint8)

        # Serialize the 6-track data using np.savez_compressed
        np.savez_compressed(file_path, six_track_data=six_track_data, length=len(self))

    def load_6_track_encoding_npz(
        self,
        directory,
        array_key="six_track_data",
        pad_length_to=0,
        zero_mean=False,
        zero_pad=True,
        channels_last=False,
        mask_percentage=0.0,
    ):
        """
        Load and process six-track encoding from a .npz file.

        Args:
            directory (str): Directory containing the .npz file.
            array_key (str, optional): Key under which the six-track data
                was saved. Defaults to 'six_track_data'.
            pad_length_to (int, optional): Length to which sequence should be padded.
                If 0, no additional padding is applied. Defaults to 0.
            zero_mean (bool, optional): If True, subtract 0.25 from each one-hot channel.
                Defaults to False.
            zero_pad (bool, optional): If True, use zeros for padding; if False, use 0.25.
                Defaults to True.
            channels_last (bool, optional): If True, channels dimension is last.
                Defaults to False.
            mask_percentage (float, optional): Percentage of data to randomly mask.
                Defaults to 0.0.

        Returns:
            np.ndarray: The loaded and processed six-track data.
        """
        # Construct the file path using gene and transcript_id
        file_path = os.path.join(
            directory,
            f"{self.gene.replace('/', '_')}_"
            f"{self.transcript_id.replace('/', '_')}_"
            f"channels_last_{channels_last}.npz",
        )

        # Load the data from the .npz file
        with np.load(file_path) as data:
            binary_data = data[array_key]

        if zero_mean:
            binary_data = binary_data.astype(np.float16)
            binary_data = binary_data - 0.25

        if mask_percentage > 0:
            # Determine the dimension to apply the mask based on channels_last
            mask_dim = 0 if channels_last else 1

            # Calculate the number of elements to mask in the selected dimension
            num_elements_to_mask = int(binary_data.shape[mask_dim] * mask_percentage)

            # Generate random indices along the selected dimension for masking
            mask_indices = np.random.choice(
                binary_data.shape[mask_dim], num_elements_to_mask, replace=False
            )

            # Apply the mask
            if channels_last:
                binary_data[mask_indices, :] = 0  # Apply mask along the 0th dimension
            else:
                binary_data[:, mask_indices] = 0  # Apply mask along the 1st dimension

        # Process the loaded data based on pad_length_to, zero_mean, and zero_pad
        if len(binary_data) < pad_length_to:
            if channels_last:
                pad_sequence = np.zeros(
                    (pad_length_to - len(binary_data), binary_data.shape[1]),
                    dtype=np.uint8,
                )
                binary_data = np.vstack(
                    (binary_data, pad_sequence)
                )  # Append the padding to the original data
            else:
                pad_sequence = np.zeros(
                    (binary_data.shape[0], pad_length_to - binary_data.shape[1]),
                    dtype=np.uint8,
                )
                binary_data = np.hstack(
                    (binary_data, pad_sequence)
                )  # Append the padding to the original data

            if not zero_pad:
                pad_sequence = pad_sequence.astype(np.float16) + 0.25

        # Optionally, apply zero-mean normalization

        return binary_data

    def check_serialized_version_exists(self, directory, channels_last=False):
        """
        Check if a serialized six-track .npz file already exists for this transcript.

        Args:
            directory (str): Directory to check for the file.
            channels_last (bool, optional): Whether the channels dimension is last.
                Defaults to False.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        # Construct the expected file path
        file_path = os.path.join(
            directory,
            f"{self.gene}_{self.transcript_id}_channels_last_{channels_last}.npz",
        )

        # Check if the file exists at the path
        return os.path.exists(file_path)


class RefseqDataset:
    """
    A dataset class that stores a collection of Transcript objects,
    providing utilities for bulk encoding and dataset operations.
    """

    def __init__(self, transcript_list: List[Transcript]):
        assert all(
            [isinstance(x, Transcript) for x in transcript_list]
        ), "Not transcripts passed into dataset {}".format(
            pd.value_counts([type(x) for x in transcript_list])
        )
        self.transcripts = transcript_list
        self.max_transcript_length = np.max([len(t) for t in self.transcripts])
        self.valid_chromosomes = ["chr{}".format(i) for i in range(1, 23)]

    def __len__(self):
        """
        Return the number of transcripts in this dataset.

        Returns:
            int: Number of Transcript objects.
        """
        return len(self.transcripts)

    def __getitem__(self, idx):
        """
        Get a transcript by index.

        Args:
            idx (int): Index in the dataset.

        Returns:
            Transcript: The Transcript object at the specified index.
        """
        return self.transcripts[idx]

    @classmethod
    def load_refseq_as_df(
        cls,
        refseq_path: str,
        mini: bool,
        chromosomes_to_use: list[str] | None,
        drop_non_nm: bool = False,
    ) -> pd.DataFrame:
        """
        Load a RefSeq annotation file into a pandas DataFrame.

        Args:
            refseq_path (str): Path to RefSeq annotation file (CSV/TSV).
            mini (bool): If True, load a small subset (1000 lines) of the file.
            chromosomes_to_use (list[str] | None): A list of chromosomes to keep,
                or None to keep all chromosomes.
            drop_non_nm (bool, optional): If True, removes non-coding transcripts
                (i.e., those whose names don't start with 'NM'). Defaults to False.

        Returns:
            pd.DataFrame: The loaded DataFrame, potentially filtered by chromosome
                and transcript type.
        """
        if mini:
            df = pd.read_csv(refseq_path, compression="infer", sep="\t", nrows=1000)
        else:
            df = pd.read_csv(refseq_path, compression="infer", sep="\t")

        # Subset to just transcript coding transcripts
        if drop_non_nm:
            df = df[df["name"].str.startswith("NM")]

        if chromosomes_to_use is not None:
            df = df[df["chrom"].isin(chromosomes_to_use)]

        return df

    @classmethod
    def refseq_df_to_transcripts(
        cls,
        df: pd.DataFrame,
        expand_transcript_distance: int,
        expand_exon_distance: int,
        genome: Genome,
    ) -> list[Transcript]:
        """
        Convert a RefSeq annotation DataFrame into a list of Transcript objects.

        Args:
            df (pd.DataFrame): DataFrame containing RefSeq annotations.
            expand_transcript_distance (int): Distance to expand upstream/downstream
                of the transcript.
            expand_exon_distance (int): Distance to expand around each exon.
            genome (Genome): Genome object providing sequence/encoding methods.

        Returns:
            list[Transcript]: A list of initialized Transcript objects.
        """
        transcripts = []
        for index, row in df.iterrows():
            exon_starts = [int(x) for x in row["exonStarts"].split(",") if x]
            exon_ends = [int(x) for x in row["exonEnds"].split(",") if x]

            # check if row has name2 column
            gene = row["name2"] if "name2" in row else "unknown"

            transcripts.append(
                Transcript(
                    genome=genome,
                    transcript_id=row["name"],
                    gene=gene,
                    exon_starts=exon_starts,
                    exon_ends=exon_ends,
                    exon_count=row["exonCount"],
                    strand=row["strand"],
                    chromosome=row["chrom"],
                    tx_start=row["txStart"],
                    tx_end=row["txEnd"],
                    cds_start=row["cdsStart"],
                    cds_end=row["cdsEnd"],
                    expand_transcript_distance=expand_transcript_distance,
                    expand_exon_distance=expand_exon_distance,
                )
            )
        return transcripts

    @classmethod
    def load_refseq(
        cls,
        refseq_path: str,
        genome: Genome,
        expand_transcript_distance: int = 0,
        expand_exon_distance: int = 0,
        mini: bool = False,
        drop_non_nm: bool = False,
        use_human_chrs: bool = True,
    ) -> list[Transcript]:
        """
        Load transcripts from a RefSeq annotation file into a list of Transcript objects.

        Args:
            refseq_path (str): Path to the RefSeq annotation CSV/TSV.
            genome (Genome): Genome object providing sequence/encoding methods.
            expand_transcript_distance (int, optional): Distance to expand upstream/downstream
                of the transcript. Defaults to 0.
            expand_exon_distance (int, optional): Distance to expand around each exon.
                Defaults to 0.
            mini (bool, optional): If True, loads only a small subset (1000 lines). Defaults to False.
            drop_non_nm (bool, optional): If True, only coding transcripts (names starting 'NM') are used.
                Defaults to False.
            use_human_chrs (bool, optional): If True, only chromosomes 'chr1' through 'chr22' and 'chrX'
                (23 total) are used. Defaults to True.

        Returns:
            list[Transcript]: The loaded Transcript objects.
        """
        if use_human_chrs:
            chromosomes_to_use = ["chr{}".format(i) for i in range(23)]
        else:
            chromosomes_to_use = None

        df = RefseqDataset.load_refseq_as_df(
            refseq_path,
            mini,
            chromosomes_to_use=chromosomes_to_use,
            drop_non_nm=drop_non_nm,
        )

        transcripts = RefseqDataset.refseq_df_to_transcripts(
            df, expand_transcript_distance, expand_exon_distance, genome
        )
        return transcripts

    def one_hot_encode_dataset(
        self,
        pad_length_to: int = 0,
        zero_mean: bool = True,
        split_transcript: int = 0,
    ) -> np.array:
        """
        One-hot-encode all transcripts in the dataset.

        Args:
            pad_length_to (int, optional): If > 0, all transcripts will be padded
                to this length. Defaults to 0, meaning no padding.
            zero_mean (bool, optional): If True, subtract 0.25 from each one-hot channel.
                Defaults to True.
            split_transcript (int, optional): If > 0, each transcript is split into
                chunks of size `split_transcript` after encoding. Defaults to 0.

        Returns:
            np.ndarray: A NumPy array of shape (N, L, 4) if not split, where N is the
                number of transcripts (or chunks if split) and L is `pad_length_to`
                or the transcript length. If split, shape is (M, split_transcript, 4),
                where M is the total number of chunks.
        """
        # if pad_length_to is not set set it to the maximum length of the transcript
        if not pad_length_to:
            pad_length_to = self.max_transcript_length

        assert (
            pad_length_to >= self.max_transcript_length
        ), "Maximum transcript length in dataset:{} greater than pad_length_to:{}".format(
            self.max_transcript_length, pad_length_to
        )
        if not split_transcript:
            padded_dataset = np.array(
                [
                    t.one_hot_encode_transcript(pad_length_to, zero_mean=zero_mean)
                    for t in self.transcripts
                ]
            )
        else:
            padded_dataset = []
            for t in self.transcripts:
                # One hot encode transcript
                t_one_hot = t.one_hot_encode_transcript(zero_mean=zero_mean)

                # Split the transcript along 0 dimension into n chunks of less than
                # split_transcript_lenght
                extend_distance = (
                    split_transcript - t_one_hot.shape[0] % split_transcript
                )

                # Distance to extend
                n_to_extend = np.zeros((extend_distance, 4), dtype=np.float32)
                t_one_hot = np.concatenate([t_one_hot, n_to_extend])

                # Split the transcript along 0 dimension into n chunks of less than
                # split_transcript_lenght
                number_of_chunks = t_one_hot.shape[0] / split_transcript
                t_one_hot = np.split(t_one_hot, number_of_chunks)

                padded_dataset += t_one_hot

            padded_dataset = np.array(padded_dataset)
        return padded_dataset

    def get_sequence_dataset(self, pad_length_to: int = 0) -> np.array:
        """
        Get the concatenated nucleotide sequences of all transcripts in the dataset.

        Args:
            pad_length_to (int, optional): If > 0, pad sequences to this length
                with 'N'. Defaults to 0.

        Returns:
            np.ndarray: An array of shape (N,) where each element is the padded
                nucleotide sequence string of a transcript.
        """
        if not pad_length_to:
            pad_length_to = self.max_transcript_length

        assert (
            pad_length_to >= self.max_transcript_length
        ), "Maximum transcript length in dataset:{} greater than pad_length_to:{}".format(
            self.max_transcript_length, pad_length_to
        )
        padded_dataset = np.array(
            [t.get_sequence(pad_length_to) for t in self.transcripts]
        )
        return padded_dataset

    def get_encoded_dataset(
        self, pad_length_to: int = 0, split_transcript: int = 0
    ) -> np.array:
        """
        Get the numeric-encoded dataset (A->0, C->1, T->2, G->3, N->4).

        Args:
            pad_length_to (int, optional): If > 0, pad each transcript to this length.
                Defaults to 0.
            split_transcript (int, optional): If > 0, split each transcript
                into chunks of this size. Defaults to 0.

        Returns:
            np.ndarray: Encoded dataset array. Shape depends on splitting:
                - Not split: (N, pad_length_to)
                - Split: (M, split_transcript), where M is the total number of chunks.
        """
        if not pad_length_to:
            pad_length_to = self.max_transcript_length

        assert (
            pad_length_to >= self.max_transcript_length
        ), "Maximum transcript length in dataset:{} greater than pad_length_to:{}".format(
            self.max_transcript_length, pad_length_to
        )

        if not split_transcript:
            padded_dataset = np.array(
                [t.encode(pad_length_to) for t in self.transcripts]
            )

        else:
            padded_dataset = []
            for t in self.transcripts:
                # encode transcript
                t_encoded = t.encode()

                extend_distance = (
                    split_transcript - t_encoded.shape[0] % split_transcript
                )

                # Distance to extend
                # Ns are indicated by 4s
                n_to_extend = np.zeros(extend_distance, dtype=int) + 4
                t_encoded = np.concatenate([t_encoded, n_to_extend])

                # Split the transcript along 0 dimension into n chunks of less than
                # split_transcript_lenght
                number_of_chunks = t_encoded.shape[0] / split_transcript
                t_encoded = np.split(t_encoded, number_of_chunks)
                padded_dataset += t_encoded

            padded_dataset = np.array(padded_dataset)

        return padded_dataset

    def drop_long_transcripts(self, max_length: int):
        """
        Drop transcripts that exceed a certain length.
        """
        # count number of long transcripts
        number_of_long_transcripts = len(
            [x for x in self.transcripts if len(x) > max_length]
        )
        self.transcripts = [x for x in self.transcripts if len(x) <= max_length]
        self.max_transcript_length = np.max([len(t) for t in self.transcripts])

        if number_of_long_transcripts > 0:
            print(
                "Dropped {} exceeding length {}".format(
                    number_of_long_transcripts, max_length
                )
            )

    def transcript_lengths(self):
        """
        Get descriptive statistics for lengths of transcripts in the dataset.

        Returns:
            dict: A dictionary containing pandas descriptive stats
                  (e.g., mean, std, min, max, etc.) for transcript lengths.
        """
        return dict(pd.Series([len(x) for x in self.transcripts]).describe())
