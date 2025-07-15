import pandas as pd


def get_ortholog_id(gene_pred_row: pd.Series, assembly_name: str) -> str:
    """Convert gene pred row for ortholog alignment into standard id.

    Alignments are assumed to come from TOGA. Thus, the sequence id has the
    format ensembl_id.gene_name.alignment_chain.

    We represent each aligned transcript using the format:

        {EnsemblID}_{AssemblyName}_{AlignmentChain}

    Args:
        gene_pred_row: Row of gene pred annotation from TOGA.
        assembly_name: Name of assembly for aligned species.

    Returns:
        Standardized representation of toga alignment.
    """
    ensembl_id = gene_pred_row.iloc[0].split(".")[0]
    gene_name = gene_pred_row.iloc[0].split(".")[1]
    toga_chain = gene_pred_row.iloc[0].split(".")[2]

    assert len(gene_name) > 0
    assert len(toga_chain) > 0

    transcript_id = "{}_{}_{}".format(
        ensembl_id,
        assembly_name,
        toga_chain,
    )

    return transcript_id


def get_augustus_ortholog_id(
    gene_pred_row: pd.Series,
    assembly_name: str
) -> str:
    """Convert gene pred row for AUGUSTUS prediction into standard id.

    Alignments come from TOGA cds alignments that are extended using AUGUSTUS
    to include untranslated regions. The sequence id has the format
    ensembl_id.gene_name.alignment_chain.

    We represent each aligned transcript using the format:

        {EnsemblID}_{AssemblyName}_{AlignmentChain}_AUGUSTUS

    Args:
        gene_pred_row: Row of gene pred annotation from AUGUSTUS.
        assembly_name: Name of assembly for aligned species.

    Returns:
        Standardized representation of AUGUSTUS-extended alignment.
    """
    ensembl_id = gene_pred_row.iloc[0].split(".")[0]
    gene_name = gene_pred_row.iloc[0].split(".")[1]
    toga_chain = gene_pred_row.iloc[0].split(".")[2]

    assert len(gene_name) > 0
    assert len(toga_chain) > 0

    transcript_id = "{}_{}_{}_AUGUSTUS".format(
        ensembl_id,
        assembly_name,
        toga_chain,
    )

    return transcript_id

def get_transcript_path(transcript_id: str) -> str:
    """Convert transcript id to save path broken by folders.

    Used to prevent too many files from accumulating in a single directory.
    Breaks down transcript into mostly 3 character segments, inspired by
    the NCBI FTP file system. Ensembl transcript IDs always consist of
    a block of letters follow by a unique 11 digit number.

    For example, a transcript with ID:

        ENST00000646410_HLcanFam4_123

    will be split into the following path:

        ENST/00/000/646/410

    This method is currently also hacked together to handle NCBI style ids.

    Args:
        transcript_id: ID of transcript to be saved.

    Returns:
        Relative path to save transcript file.
    """
    root_id = transcript_id.split("_")[0]

    if len(root_id) == 2:
        # NCBI style ID
        species_id = root_id
        numeric_id = transcript_id.split(".")[0].split("_")[1]

        chunks = [numeric_id[i:i + 3] for i in range(0, len(numeric_id), 3)]
        path = root_id + "/" + "/".join(chunks)
    elif len(root_id) == 3:
        # Kaessmann style ID
        species_id = "_".join(transcript_id.split("_")[:2])

        numeric_id = transcript_id.split("_")[2]

        path = "{}/{}/{}/{}".format(
            species_id,
            numeric_id[:4],
            numeric_id[4:6],
            numeric_id[6:8],
        )

    else:
        # Ensembl style ID
        species_id = "".join([c for c in root_id if not c.isdigit()])
        ensembl_id = "".join([c for c in root_id if c.isdigit()])

        path = "{}/{}/{}/{}/{}".format(
            species_id,
            ensembl_id[:2],
            ensembl_id[2:5],
            ensembl_id[5:8],
            ensembl_id[8:11],
        )

    return path


def augustus_name_fn(assembly_path: str, row: pd.Series) -> tuple[str, str]:
    toga_gene_name = row["name"].split(".")[1]
    assembly_name = assembly_path.split("/")[-1].split(".")[0]
    transcript_id = get_augustus_ortholog_id(row, assembly_name)

    toga_gene_name = str(toga_gene_name).lower()

    return transcript_id, toga_gene_name


def ortho_name_fn(assembly_path: str, row: pd.Series) -> tuple[str, str]:
    toga_gene_name = row["name"].split(".")[1]
    assembly_name = assembly_path.split("/")[-1].split(".")[0]
    transcript_id = get_ortholog_id(row, assembly_name)

    toga_gene_name = str(toga_gene_name).lower()

    return transcript_id, toga_gene_name


def ref_name_fn(assembly_path: str, row: pd.Series) -> tuple[str, str]:
    transcript_id = row["name"].split(".")[0]
    gene_name = row["name2"] if "name2" in row else "unknown"

    gene_name = str(gene_name).lower()

    return transcript_id, gene_name
