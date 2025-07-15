import os
from pathlib import Path

__all__ = ["Interval", "Transcript", "RefseqDataset"]

import pandas as pd
import numpy as np
from orthrus.genome import Genome
from typing import List
import random 

def mask_encoding(data, mask_percentage: float, channels_last: bool = False):
    """
    Mask a percentage of the encoding data by setting it to zero.
    This is useful if we don't have enough splice isoforms/orthologs/pre-mRNA
    to sample from.
    :param data: The encoding data to be masked.
    :param mask_percentage: The percentage of the data to be masked.
    :return: The masked encoding data.
    """
    # if data is a view, we need to copy it to avoid modifying the original data
    data = np.array(data, copy=True)

    if channels_last:
        data = np.swapaxes(data, 0, 1)

    num_elements_to_mask = int(data.shape[1] * mask_percentage)
    mask_indices = np.random.choice(
        data.shape[1],
        num_elements_to_mask,
        replace=False
    )
    data[:, mask_indices] = 0

    if channels_last:
        data = np.swapaxes(data, 0, 1)

    return data

class SpeciesGene:
    def __init__(self, pre_mrna, splice_isoforms, orthologs):
        self.pre_mrna = pre_mrna
        self.splice_isoforms = splice_isoforms
        self.orthologs = orthologs

    def sample_splice_isoforms(self, n: int = 4, masking_prop: float = 0.0):
        """
        Sample n splice isoforms from the gene.
        """

        sampled = []

        sampled.extend(self.splice_isoforms)

        if n > len(self.splice_isoforms):

            additional_samples = n - len(self.splice_isoforms)

            sampled.extend(random.choices(self.splice_isoforms, k=additional_samples))

        elif len(self.splice_isoforms) > 0:
            sampled.extend(random.sample(self.splice_isoforms, n))

        if masking_prop > 0:
            # Mask the samples
            for i in range(len(sampled)):
                sampled[i] = mask_encoding(
                    sampled[i], masking_prop
                )
        
        return sampled

    def sample_orthologs(self, n: int = 4, masking_prop: float = 0.0):
        """
        Sample n orthologs from the gene.
        """
        
        sampled = []

        sampled.extend(self.orthologs)

        if n > len(self.orthologs):

            additional_samples = n - len(self.orthologs)

            sampled.extend(random.choices(self.orthologs, k=additional_samples))

        elif len(self.orthologs) > 0:
            sampled.extend(random.sample(self.orthologs, n))

        if masking_prop > 0:
            # Mask the samples
            for i in range(len(sampled)):
                sampled[i] = mask_encoding(
                    sampled[i], masking_prop
                )
        
        return sampled

    def sample_locals(self, n: int = 8, masking_prop: float = 0.0, iso_weight: float = 0.5):
        """
        Sample n local sequences from the splice isoforms and orthologs.
        If one of the lists is empty, sample all n from the other.
        Otherwise, sample int(n * iso_weight) from splice isoforms and n - int(n * iso_weight) from orthologs.
        """
        # If both lists are empty, return an empty list.
        if not self.splice_isoforms and not self.orthologs:
            return []
        # If one of the lists is empty, sample all from the other.
        if not self.splice_isoforms:
            return self.sample_orthologs(n)
        if not self.orthologs:
            return self.sample_splice_isoforms(n)

        # Determine desired counts using iso_weight.
        n_iso = int(n * iso_weight)
        n_ortho = n - n_iso

        splice_samples = self.sample_splice_isoforms(n_iso, masking_prop)
        ortholog_samples = self.sample_orthologs(n_ortho, masking_prop)
        return splice_samples + ortholog_samples

class Gene:
    def __init__(self):
        self.species_genes = {}  # keys: species name, values: SpeciesGene objects
    
    def sample_species(self, n: int = 2):
        """
        Sample n species from the gene.
        """
    
        if len(self.species_genes) == 0:
            return []
        if n > len(self.species_genes):
            additional_samples = n - len(self.species_genes)
            replacement_samples = random.choices(list(self.species_genes.keys()), k=additional_samples)
            return list(self.species_genes.keys()) + replacement_samples
        else:
            return random.sample(list(self.species_genes.keys()), n)

    def sample_globals(self, species: list, masking_prop: float = 0.0):
        """
        Get the pre-mRNA sequences for each of the given species and add masking.
        """

        sampled = []

        for species_name in species:
            if species_name in self.species_genes:
                pre_mrna = self.species_genes[species_name].pre_mrna
                if masking_prop > 0:
                    pre_mrna = mask_encoding(pre_mrna, masking_prop)
                sampled.append(pre_mrna)

        return sampled

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
        return self.end - self.start

    def __repr__(self):
        """
        Return string representation of a transcript
        """
        return "Interval {}:{}-{}:{}".format(
            self.chromosome, self.start, self.end, self.strand
        )

    def overlaps(self, interval):

        assert type(interval) == Interval
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
        assert type(interval) == Interval
        if interval.chrom != self.chrom:
            return False
        if interval.strand != self.strand:
            return False

        after_start = self.start >= interval.start
        before_end = self.end <= interval.end

        return after_start and before_end

    def one_hot_encode(self, zero_mean: bool = True):
        seq = self.genome.get_encoding_from_coords(
            self.chrom, self.start, self.end, self.strand
        )
        if zero_mean:
            seq = seq - 0.25
        return seq

    def sequence(self):
        return self.genome.get_sequence_from_coords(
            self.chrom, self.start, self.end, self.strand
        )

    def encode(self):
        seq = self.genome.get_sequence_from_coords(
            self.chrom, self.start, self.end, self.strand
        ).upper()
        return np.array([self.alphabet_map[x] for x in seq])


class Transcript(object):
    """
    An object reprenting an RNA transcript allowing to query RNA sequence and
    convert to one hot encoding.
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
        transcript_id
        gene
        exon_starts
        exon_ends
        genome
        exon_count
        pad transcript
        expand_transcript_distance
        expand_exon_distance
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
        Length returns the total length of the generated sequence including
        exon expand distance and transcript expand distance.
        """
        return np.sum([len(x) for x in self.transcript_intervals])

    def __repr__(self) -> str:
        """Return string representation of a transcript."""
        return "Transcript {} {}:{}-{}:{}".format(
            self.transcript_id,
            self.chromosome,
            self.tx_start,
            self.tx_end,
            self.strand
        )

    def __hash__(self):
        identifier = (
            f"{self.gene} length {self.__len__()} n_exons {self.exon_count}"
            f"{self.transcript_id} {self.chromosome}:{self.tx_start}-{self.tx_end}:{self.strand}"
        )
        return hash(identifier)

    def __eq__(self, other):
        gene_eq = self.gene == other.gene
        len_eq = self.__len__() == other.__len__()
        exn_cnt_eq = self.exon_count == other.exon_count
        trsc_id_eq = self.transcript_id == other.transcript_id
        chrom_eq = self.chromosome == other.chromosome
        tx_start_eq = self.tx_start == other.tx_start
        tx_end_eq = self.tx_end == other.tx_end
        stand_eq = self.strand == other.strand
        conditions = [gene_eq, len_eq, exn_cnt_eq, trsc_id_eq, chrom_eq, tx_start_eq, tx_end_eq, stand_eq]
        return all(conditions)

    def generate_inter_exon_distances(self):
        """
        Generates distances between exons
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
        """Calculates the expand distance for every single exon.

        Makes sure there are no overlapping sequences.
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
            ), "Length of transcript {} greater than padding specified {}".format(
                len(self), pad_length_to
            )
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
            'TTT': 'F', 'CTT': 'L', 'ATT': 'I', 'GTT': 'V',
            'TTC': 'F', 'CTC': 'L', 'ATC': 'I', 'GTC': 'V',
            'TTA': 'L', 'CTA': 'L', 'ATA': 'I', 'GTA': 'V',
            'TTG': 'L', 'CTG': 'L', 'ATG': 'M', 'GTG': 'V',
            'TCT': 'S', 'CCT': 'P', 'ACT': 'T', 'GCT': 'A',
            'TCC': 'S', 'CCC': 'P', 'ACC': 'T', 'GCC': 'A',
            'TCA': 'S', 'CCA': 'P', 'ACA': 'T', 'GCA': 'A',
            'TCG': 'S', 'CCG': 'P', 'ACG': 'T', 'GCG': 'A',
            'TAT': 'Y', 'CAT': 'H', 'AAT': 'N', 'GAT': 'D',
            'TAC': 'Y', 'CAC': 'H', 'AAC': 'N', 'GAC': 'D',
            'TAA': '*', 'CAA': 'Q', 'AAA': 'K', 'GAA': 'E',
            'TAG': '*', 'CAG': 'Q', 'AAG': 'K', 'GAG': 'E',
            'TGT': 'C', 'CGT': 'R', 'AGT': 'S', 'GGT': 'G',
            'TGC': 'C', 'CGC': 'R', 'AGC': 'S', 'GGC': 'G',
            'TGA': '*', 'CGA': 'R', 'AGA': 'R', 'GGA': 'G',
            'TGG': 'W', 'CGG': 'R', 'AGG': 'R', 'GGG': 'G'
        }

        protein_sequence = ''
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i + 3].upper()
            protein_sequence += codon_map.get(codon, '?')

        return protein_sequence

    def get_amino_acid_sequence(self):
        rel_cds_start, rel_cds_end = self.calculate_relative_cds_start_end()
        nt_sequence = self.get_sequence()
        coding_sequence = nt_sequence[rel_cds_start: rel_cds_end]
        aa_seq = Transcript.translate_dna(coding_sequence)
        return aa_seq

    def encode(self, pad_length_to: int = 0):
        if self.strand == "+":
            seqs = np.concatenate(
                [x.encode() for x in self.transcript_intervals]
            ).flatten()
        elif self.strand == "-":
            seqs = np.concatenate(
                [x.encode() for x in self.transcript_intervals[::-1]]
            ).flatten()

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
        rel_ss = self.calculate_relative_splice_sites()

        if pad_length_to == 0:
            encoding_length = len(self)
        else:
            encoding_length = pad_length_to

        ss_encoded = np.zeros(encoding_length)
        ss_encoded[rel_ss] = 1

        return ss_encoded.reshape(-1, 1)

    def encode_coding_sequence_track(self, pad_length_to: int = 0):

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
        channels_last: bool = False
    ):
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

        save_name = "{}.npz".format(self.transcript_id)

        # Generate the file path based on gene and transcript_id
        file_path = dir_path / save_name

        if os.path.exists(file_path) and skip_existing:
            return

        # Encode the 6-track data
        six_track_data = self.encode_6_track(
            pad_length_to=0,
            zero_mean=False,
            zero_pad=True,
            channels_last=False
        ).astype(np.uint8)

        # Serialize the 6-track data using np.savez_compressed
        np.savez_compressed(
            file_path,
            six_track_data=six_track_data,
            length=len(self)
        )

    def load_6_track_encoding_npz(
        self,
        directory,
        array_key='six_track_data',
        pad_length_to=0,
        zero_mean=False,
        zero_pad=True,
        channels_last=False,
        mask_percentage=0.0,
    ):
        """
        Loads and processes binary data saved as uint8 from a .npz file based on padding and normalization parameters.

        :param directory: The directory from which to load the .npz file.
        :param array_key: The key that was used to save the binary data in the .npz file.
        :param pad_length_to: The length to which the sequence should be padded.
        :param zero_mean: Whether the data should be zero-mean normalized.
        :param zero_pad: Whether the padding should use zeros or a constant value.
        :param mask_percentage: The percentage of the data to randomly mask.
        :return: A NumPy array of the processed binary data.
        """
        # Construct the file path using gene and transcript_id
        file_path = os.path.join(
            directory,
            f"{self.gene.replace('/', '_')}_"
            f"{self.transcript_id.replace('/', '_')}_"
            f"channels_last_{channels_last}.npz"
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
            mask_indices = np.random.choice(binary_data.shape[mask_dim], num_elements_to_mask, replace=False)

            # Apply the mask
            if channels_last:
                binary_data[mask_indices, :] = 0  # Apply mask along the 0th dimension
            else:
                binary_data[:, mask_indices] = 0  # Apply mask along the 1st dimension

        # Process the loaded data based on pad_length_to, zero_mean, and zero_pad
        if len(binary_data) < pad_length_to:
            if channels_last:
                pad_sequence = np.zeros((pad_length_to - len(binary_data), binary_data.shape[1]), dtype=np.uint8)
                binary_data = np.vstack((binary_data, pad_sequence))  # Append the padding to the original data
            else:
                pad_sequence = np.zeros((binary_data.shape[0], pad_length_to - binary_data.shape[1]), dtype=np.uint8)
                binary_data = np.hstack((binary_data, pad_sequence))  # Append the padding to the original data

            if zero_pad == False:
                pad_sequence = pad_sequence.astype(np.float16) + 0.25

        # Optionally, apply zero-mean normalization

        return binary_data

    def check_serialized_version_exists(self, directory, channels_last=False):
        """
        Checks if there is a serialized version of the sequence available in the specified directory.

        :param directory: The directory to check for the serialized file.
        :return: True if the serialized file exists, False otherwise.
        """
        # Construct the expected file path
        file_path = os.path.join(directory, f"{self.gene}_{self.transcript_id}_channels_last_{channels_last}.npz")

        # Check if the file exists at the path
        return os.path.exists(file_path)


# %% ../nbs/data.ipynb 8
class RefseqDataset:
    """Refseq dataset."""

    def __init__(self, transcript_list: List[Transcript]):
        assert all(
            [type(x) == Transcript for x in transcript_list]
        ), "Not transcripts passed into dataset {}".format(
            pd.value_counts([type(x) for x in transcript_list])
        )
        self.transcripts = transcript_list
        self.max_transcript_length = np.max([len(t) for t in self.transcripts])
        self.valid_chromosomes = ["chr{}".format(i) for i in range(23)]

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        return self.transcripts[idx]

    @classmethod
    def load_refseq_as_df(
        cls,
        refseq_path: str,
        mini: bool,
        chromosomes_to_use: list[str] | None,
        drop_non_nm: bool = False,
    ) -> pd.DataFrame:
        """Load RefSeq annotation file into pandas dataframe.

        Args:
            refseq_path: Path to refseq annotation.
            mini: Loads subset of all annootations.
            chromosomes_to_use:
                Only annotations from chromosomes in this list are kept. Set
                to None if no filtering is desired.
            drop_non_nm: Removes non-coding transcripts.

        Returns:
            Dataframe of all filtered anotations from refseq file.
        """
        if mini:
            df = pd.read_csv(
                refseq_path,
                compression="infer",
                sep="\t",
                nrows=1000
            )
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
        genome: Genome
    ) -> list[Transcript]:
        """Initialize Transcript for each annotation in refseq df.

        Args:
            df: DataFrame containing refseq annotations.
            expand_transcript_distance:
            expand_exon_distance:
            genome: Genome associated with annotation.

        Returns:
            List of Transcript objects.
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
        use_human_chrs: bool = True
    ) -> list[Transcript]:
        """Load transcripts from refseq annotation.

        Args:
            refseq_path: Path to refseq annotation csv.
            genome: Genome associated with annotations.
            expand_transcript_distance:
            expand_exon_distance:
            mini: Selects subset of annotations to use.
            drop_non_nm: Selects only coding transcripts.
            use_human_chrs: Selects transcripts from only first 23 chrs.

        Returns:
            List of transcripts loaded from refseq annotations.
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
            df,
            expand_transcript_distance,
            expand_exon_distance,
            genome
        )
        return transcripts

    def one_hot_encode_dataset(
        self,
        pad_length_to: int = 0,
        zero_mean: bool = True,
        split_transcript: int = 0,
    ) -> np.array:

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
        return dict(pd.Series([len(x) for x in self.transcripts]).describe())
