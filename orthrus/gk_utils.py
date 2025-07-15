import numpy as np


def find_transcript(genome, transcript_id):
    """Find a transcript in a genome by transcript ID.

    Args:
        genome (object): The genome object containing a list of transcripts.
        transcript_id (str): The ID of the transcript to find.

    Returns:
        object: The transcript object, if found.

    Raises:
        ValueError: If no transcript with the given ID is found.

    Example:
        >>> # Create sample transcripts and a genome
        >>> transcript1 = 'ENST00000263946'
        >>> genome = Genome("gencode.v29")
        >>> result = find_transcript(genome, 'ENST00000335137')
        >>> print(result.id)
        <Transcript ENST00000263946.7 of PKP1>
        >>> # If transcript ID is not found
        >>> find_transcript(genome, 'ENST00000000000')
        ValueError: Transcript with ID ENST00000000000 not found.
    """
    transcripts = [x for x in genome.transcripts if x.id.split(".")[0] == transcript_id]
    if not transcripts:
        raise ValueError(f"Transcript with ID {transcript_id} not found.")

    return transcripts[0]


def find_transcript_by_gene_name(genome, gene_name):
    """Find all transcripts in a genome by gene name.

    Args:
        genome (object): The genome object containing a list of transcripts.
        gene_name (str): The name of the gene whose transcripts are to be found.

    Returns:
        list: A list of transcript objects corresponding to the given gene name.

    Raises:
        ValueError: If no transcripts for the given gene name are found.

    Example:
        >>> # Find transcripts by gene name
        >>> transcripts = find_transcript_by_gene_name(genome, 'PKP1')
        >>> print(transcripts)
        [<Transcript ENST00000367324.7 of PKP1>,
        <Transcript ENST00000263946.7 of PKP1>,
        <Transcript ENST00000352845.3 of PKP1>,
        <Transcript ENST00000475988.1 of PKP1>,
        <Transcript ENST00000477817.1 of PKP1>]
        >>> # If gene name is not found
        >>> find_transcript_by_gene_name(genome, 'XYZ')
        ValueError: No transcripts found for gene name XYZ.
    """
    genes = [x for x in genome.genes if x.name == gene_name]
    if not genes:
        raise ValueError(f"No genes found for gene name {gene_name}.")
    if len(genes) > 1:
        print(f"Warning: More than one gene found for gene name {gene_name}.")
        print("Concatenating transcripts from all genes.")

    transcripts = []
    for gene in genes:
        transcripts += gene.transcripts
    return transcripts


def create_cds_track(t):
    """
    Create a track for the coding sequence of a transcript, ignoring t.utr5s/t.utr3s.

    - The final track length = sum of all exon lengths.
    - The region before the CDS is zeros (the '5′ UTR').
    - The CDS region is an every-third=1 pattern.
    - The region after is zeros (the '3′ UTR').

    Args:
        t (gk.Transcript): The transcript object. Must have `t.cdss` for coding intervals.

    Returns:
        np.ndarray: 1D array of shape (transcript_length,).
                    0 for noncoding positions, 1 every third base in the CDS region.
    """
    # 1) Compute total length of the transcript (sum of exon lengths)
    transcript_length = sum(len(exon) for exon in t.exons)
    if transcript_length == 0:
        return np.array([], dtype=int)

    # 2) If there are no CDS intervals, return an all-zero track
    cds_intervals = t.cdss
    if not cds_intervals:
        return np.zeros(transcript_length, dtype=int)

    # 3) Sum the lengths of all CDS intervals
    cds_length = sum(len(c) for c in cds_intervals)

    five_utr_length = 0
    for exon in t.exons:
        if not exon.overlaps(cds_intervals[0]):
            five_utr_length += len(exon)
        else:
            # calculate the difference between 5'UTR of two exons
            cds_end5 = cds_intervals[0].end5.start
            exon_end5 = exon.end5.start
            diff = abs(cds_end5 - exon_end5)
            five_utr_length += diff
            break

    # 6) The remainder after we place the CDS is the "3′ UTR" length
    three_utr_length = transcript_length - (five_utr_length + cds_length)
    assert three_utr_length >= 0
    # 7) Build the CDS region track: every 3rd base is 1
    cds_track = np.zeros(cds_length, dtype=int)
    cds_track[0::3] = 1

    # 8) Concatenate: 5′ zeros, the CDS track, 3′ zeros
    track = np.concatenate(
        [
            np.zeros(five_utr_length, dtype=int),
            cds_track,
            np.zeros(three_utr_length, dtype=int),
        ]
    )

    return track


def create_splice_track(t):
    """Create a track of the splice sites of a transcript.
    The track is a 1D array where the positions of the splice sites are 1.

    Args:
        t (gk.Transcript): The transcript object.
    """
    len_mrna = sum([len(x) for x in t.exons])

    splicing_track = np.zeros(len_mrna, dtype=int)
    cumulative_len = 0
    for exon in t.exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1 : cumulative_len] = 1

    return splicing_track


# convert to one hot
def seq_to_oh(seq):
    oh = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq):
        if base == "A":
            oh[i, 0] = 1
        elif base == "C":
            oh[i, 1] = 1
        elif base == "G":
            oh[i, 2] = 1
        elif base == "T":
            oh[i, 3] = 1
    return oh


def create_one_hot_encoding(t, genome):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.

    Args
        t (gk.Transcript): The transcript object.
    """
    seq = "".join([genome.dna(exon) for exon in t.exons])
    oh = seq_to_oh(seq)
    return oh


def create_six_track_encoding(t, genome, channels_last=False):
    """Create a track of the sequence of a transcript.

    Produces an array of shape (L,6) if channels_last=True
    or (6,L) if channels_last=False.

    Args:
        t (gk.Transcript): The transcript object.
        genome (gk.Genome): Genome reference.
        channels_last (bool): If True, output is (L, 6). Otherwise, (6, L).

    Returns:
        np.ndarray: A 2D array with 6 channels (one-hot base encoding + CDS + splice).
    """
    # Step 1: Generate base tracks
    oh = create_one_hot_encoding(t, genome)  # shape is (L, 4)
    cds_track = create_cds_track(t)  # shape is (L,)
    splice_track = create_splice_track(t)  # shape is (L,)

    # Step 2: Create final track based on channels_last
    if channels_last:
        # Channels along axis=1 => shape (L, 6)
        # (L, 4), (L, 1), (L, 1) -> (L, 6)
        six_track = np.concatenate(
            [oh, cds_track[:, None], splice_track[:, None]], axis=1
        )
    else:
        # Channels along axis=0 => shape (6, L)
        # first transpose one-hot from (L, 4) to (4, L)
        oh = oh.T
        # reshape cds/splice from (L,) to (1, L)
        cds_track = cds_track[None, :]
        splice_track = splice_track[None, :]
        # now concatenate on axis=0 => shape (6, L)
        six_track = np.concatenate([oh, cds_track, splice_track], axis=0)

    return six_track
