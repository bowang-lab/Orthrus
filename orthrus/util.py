from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import os
import random
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from absl import flags

from .data import RefseqDataset, Transcript
from .genome import Genome


@dataclass
class OrthologData:
    refseqdataset: RefseqDataset
    species_name: str
    species_pair: str


def make_timestamp():
    timestamp = "_".join(re.split(":|-| ", str(datetime.now()).split(".")[0]))
    return timestamp


def restructure_dict(
    gt_dict: dict[str, list[Transcript]]
) -> dict[Transcript, list[Transcript]]:
    """Restructure Gene-Transcript map to Transcript-Transcript map.

    Args:
        gt_dict: Mapping of Gene IDs to transcripts.

    Returns:
        Mapping of Transcripts to a list of associated Transcripts.
    """
    new_dict = {}

    for _, objs in tqdm(gt_dict.items(), total=len(gt_dict), desc="Remapping"):
        object_set = set(objs)
        for obj in object_set:
            if obj not in new_dict:
                new_dict[obj] = set()

            new_dict[obj].update(object_set - {obj})

    # Convert sets back to lists in the final dictionary
    return {key: list(value) for key, value in new_dict.items()}


def split_dict(d: dict, percent: float, seed: int = 42) -> tuple[dict, dict]:
    """
    Splits a dictionary into two dictionaries randomly based on a specified
    percentage, with an option to use a random seed for reproducibility.

    Args:
        d: The dictionary to be split.
        percent: The percent of items to be included in the first dictionary.
        seed: The random seed for reproducibility.

    Returns:
        A tuple of two dicts. The first dictionary contains the specified
        percent of items, and the second dictionary contains the rest.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Convert dictionary items to a list and shuffle
    items = list(d.items())
    random.shuffle(items)

    # Calculate the size of the first subset
    subset_size = int(len(d) * (1 - (percent / 100.0)))

    # Split the shuffled list into two parts
    subset1_items = items[:subset_size]
    subset2_items = items[subset_size:]

    # Convert the list of tuples back into dictionaries
    subset1 = dict(subset1_items)
    subset2 = dict(subset2_items)

    return subset1, subset2


def add_ortholog_to_transcript_dict(
    t_id_to_t: dict[str: Transcript],
    ortholog_t_list: list[Transcript],
    t_to_t_dict: dict[Transcript, list[Transcript]]
) -> dict[Transcript, list[Transcript]]:
    """Adds orthologs to the transcript dictionary.

    Args:
        t_id_to_t: Dict mapping transcript id to associated transcripts.
        ortholog_t_list:
            A list of orthologous transcripts. Each transcript's id
            indicates the host species matching transcript.
        t_to_t_dict: Reference species' transcript to transcript mapping.

    Returns:
        Reference species' transcript-transcript dict with orthologs added.
    """
    for transcript in ortholog_t_list:
        host_t_info = transcript.transcript_id.split('.')
        host_transcript = host_t_info[0]

        if host_transcript in t_id_to_t:
            host_transcript_obj = t_id_to_t[host_transcript]
            t_to_t_dict[host_transcript_obj].append(transcript)

    return t_to_t_dict


def load_appris(
    unique_transcripts=True,
    appris_dir: str = "/data1/morrisq/ian/rna_contrast/ref_annotations/",
    appris_file: str = "appris_data_human.principal_gencode_v41.txt"
):
    # generate doc string
    """
    Load the appris data
    :param unique_transcripts: whether to load only unique transcripts
    :return: the appris data
    """
    # ## load human appris
    app_h = pd.read_csv(f'{appris_dir}/{appris_file}', sep='\t')
    print(app_h['Gene ID'].duplicated().sum())
    app_h['numeric_value'] = app_h['APPRIS Annotation'].str.split(':').str[1]
    app_h['key_value'] = app_h['APPRIS Annotation'].str.split(':').str[0]
    app_h = app_h.sort_values(
        ['Gene ID', 'key_value', 'numeric_value', "Transcript ID"],
        ascending=[True, False, True, True],
    )
    if unique_transcripts:
        app_h = app_h[~app_h.duplicated('Gene ID')]
        app_h = app_h[~app_h.duplicated('Gene name')]
    return app_h


def construct_human_mouse_transcript_dict(
    transcript_length_drop=12288,
    refseq_location_human="../data/gencode_basic_v41.tsv",
    refseq_location_mouse="../data/wgEncodeGencodeBaseicVM25.tsv",
    fasta_file_location_human="../data/hg38.fa",
    fasta_file_location_mouse="../data/mm10.fa",
    mini_dataset=False,
    do_homolog_map=False,
    drop_non_nm=False,
):
    """
    Construct a transcript dictionary for human and mouse
    :param transcript_length_drop: the length to drop
    :param refseq_location_human: the location of the human refseq
    :param refseq_location_mouse: the location of the mouse refseq
    :param fasta_file_location_human: the location of the human fasta file
    :param fasta_file_location_mouse: the location of the mouse fasta file
    :param mini_dataset: whether to use a mini dataset
    :param do_homolog_map: whether to do a homolog map
    :param drop_non_nm: whether to drop non nm
    :return: the transcript dictionary
    """
    # make sure that at least one of those is present
    assert bool(refseq_location_human) + bool(refseq_location_mouse) >= 1
    all_transcripts = []
    if refseq_location_human:
        refseq_data_human = prepare_refseq_dataset(
            transcript_length_drop=transcript_length_drop,
            refseq_path=refseq_location_human,
            fasta_path=fasta_file_location_human,
            mini_dataset=mini_dataset,
            drop_non_nm=drop_non_nm,
        )
        if do_homolog_map:
            refseq_data_human = rename_gene_names_using_homology_map(refseq_data_human)

        all_transcripts.extend(refseq_data_human.transcripts)

    if refseq_location_mouse:
        refseq_data_mouse = prepare_refseq_dataset(
            transcript_length_drop=transcript_length_drop,
            refseq_path=refseq_location_mouse,
            fasta_path=fasta_file_location_mouse,
            mini_dataset=mini_dataset,
            drop_non_nm=drop_non_nm,
        )
        if do_homolog_map:
            refseq_data_mouse = rename_gene_names_using_homology_map(refseq_data_mouse)

        all_transcripts.extend(refseq_data_mouse.transcripts)

    refseq_data = RefseqDataset(all_transcripts)

    transcript_dict = generate_gene_to_transcript_dict(refseq_data)
    return transcript_dict


def generate_gene_to_transcript_dict(refseq_data):
    """Generate a dictionary of gene to transcript.

    :param refseq_data: the refseq data
    :return: the dictionary of gene to transcript
    """
    transcript_dict = defaultdict(list)
    for t in refseq_data.transcripts:
        transcript_dict[t.gene].append(t)

    return transcript_dict


def prepare_refseq_dataset(
    refseq_path: str,
    fasta_path: str,
    mini_dataset: bool = False,
    drop_non_nm: bool = False,
    use_human_chrs: bool = True,
    transcript_length_drop: int = 12288,
) -> RefseqDataset:
    """Load annotations, genome, and construct dataset of transcripts.

    Args:
        refseq_path: Path to refseq annotation file.
        fasta_path: Path to fasta represention of associated genome.
        mini_dataset: Loads only a subset of transcripts into dataset.
        drop_non_nm: Loads only coding transcripts.
        use_human_chrs: Uses only first 23 chromosomes.
        transcript_length_drop: Removes transcripts exceeding this length.

    Returns:
        Dataset of transcripts.
    """
    genome = Genome(fasta_path)

    transcripts = RefseqDataset.load_refseq(
        refseq_path,
        expand_transcript_distance=0,
        expand_exon_distance=0,
        mini=mini_dataset,
        genome=genome,
        drop_non_nm=drop_non_nm,
        use_human_chrs=use_human_chrs,
    )

    refseq_data = RefseqDataset(transcripts)
    refseq_data.drop_long_transcripts(transcript_length_drop)
    return refseq_data


def pretty_print_flags():
    FLAGS = flags.FLAGS
    print("Configuration:")
    for flag_name in sorted(FLAGS):
        flag_value = FLAGS[flag_name].value
        print(f"{flag_name}: {flag_value}")


def rename_gene_names_using_homology_map(
    refseq_data,
    homolog_matching_df="/ssd005/home/phil/Documents/01_projects/contrastive_rna_representation/data/HOM_MouseHumanSequence.rpt",
):
    df = pd.read_csv(homolog_matching_df, sep="\t")

    # Create a new transcript dict with new gene names
    name_to_id = {row["Symbol"]: row["DB Class Key"] for index, row in df.iterrows()}
    new_transcripts = list()
    for transcript in refseq_data.transcripts:
        # if transcript gene is in name to id dict then change the gene name to new name
        if transcript.gene in name_to_id.keys():
            transcript.gene = name_to_id[transcript.gene]

        new_transcripts.append(transcript)
    return RefseqDataset(new_transcripts)


def construct_combination_transcript_dict(
    refseq_files: tuple[str],
    fasta_files: tuple[str],
    species_names: tuple[str],
    transcript_length_drop=12288,
    mini_dataset=False,
    drop_non_nm=False,
    do_homologene_map=False,
) -> dict[any, list]:
    """Load transcripts for all pairs of annotation and fasta files.

    Args:
        refseq_files: Paths to each refseq annotation file.
        fasta_files: Paths to each fasta associated with annotation.
        species_name: Name of annotated species.
        transcript_length_drop: Drops transcripts exceeding this length.
        mini_dataset: Selects a subset of all annotations.
        drop_non_nm: Selects only coding transcripts.
        do_homologene_map: Renames genes using homologene map.

    Returns:
        Dictionary mapping transcript gene names to associated transcripts.
    """
    assert len(refseq_files) > 1 and len(fasta_files) > 1
    assert len(fasta_files) == len(refseq_files)

    all_transcripts = []
    for r_path, f_path, spec in zip(refseq_files, fasta_files, species_names):
        print("Preparing transcripts for:", spec, flush=True)

        refseq_data = prepare_refseq_dataset(
            refseq_path=r_path,
            fasta_path=f_path,
            mini_dataset=mini_dataset,
            drop_non_nm=drop_non_nm,
            use_human_chrs=spec in ["human", "mouse"],
            transcript_length_drop=transcript_length_drop
        )

        if do_homologene_map:
            refseq_data = multi_species_homologene_map(refseq_data, spec)

        all_transcripts.extend(refseq_data.transcripts)

    refseq_data = RefseqDataset(all_transcripts)

    transcript_dict = generate_gene_to_transcript_dict(refseq_data)

    return transcript_dict


def multi_species_homologene_map(
    refseq_data,
    species_name,
    homologene_path=(
        '/scratch/hdd001/home/phil/rna_contrast/datasets/data_for_ian/annotation_data/homology_maps_homologene'
    )
):
    """
    Rename gene names using homologene map
    :param refseq_data: the refseq data
    :param species_name: the species name
    :param homologene_path: the homologene path
    :return: the refseq data with the gene names renamed
    """
    df = pd.read_csv(f"{homologene_path}/{species_name}_homology_map.csv")
    name_to_id = {
        row["gene_name"]: species_name + "_" + f"{row['gene_group']}" for _, row in df.iterrows()
    }
    new_transcripts = list()
    for transcript in refseq_data.transcripts:
        # if transcript gene is in name to id dict then change the gene name to new name
        if transcript.gene in name_to_id.keys():
            transcript.gene = name_to_id[transcript.gene]

        new_transcripts.append(transcript)
    return RefseqDataset(new_transcripts)


def load_multi_species_refseq(
    pair_file_path: str,
    refseq_path: str,
    fasta_path: str,
    mini: bool = False,
    do_homologene_map: bool = True,
    transcript_length_drop: int = 12288,
) -> dict[str: list[Transcript]]:
    """Construct Gene-Transcript map for union of specified species.

    The loaded species can be loaded from the pair file, which is a CSV with
    scheme (species_name, refseq_annotation, assembly). The annotation and
    assembly files should be kept at the specified paths.

    Args:
        pair_file_path: Path to pair file.
        refseq_path: Path to folder of refseq annotations.
        fasta_path: Path to folder of assembly fastas.
        mini: Loads of a subset of all annotations.
        do_homologene_map: Whether to do a homologene map.
        transcript_length_drop: Drops transcripts above this length.

    Returns:
        Union of all Gene-Transcripts for all specifies species.
    """
    base_genomes = pd.read_csv(pair_file_path)

    base_genomes.iloc[:, 1] = base_genomes.iloc[:, 1].apply(
        lambda x: os.path.join(refseq_path, x)
    )
    base_genomes.iloc[:, 2] = base_genomes.iloc[:, 2].apply(
        lambda x: os.path.join(fasta_path, x)
    )

    transcript_map = construct_combination_transcript_dict(
        refseq_files=base_genomes.iloc[:, 1].to_list(),
        fasta_files=base_genomes.iloc[:, 2].to_list(),
        species_names=base_genomes.iloc[:, 0].to_list(),
        transcript_length_drop=transcript_length_drop,
        mini_dataset=mini,
        do_homologene_map=do_homologene_map,
    )
    return transcript_map


def load_orthologs_by_path(
    pair_file_path: str,
    transcript_length_drop: int = 12288,
    mini: bool = True
) -> list[OrthologData]:
    """Load orthologs using pairing file generated by Zoonomia Downloader.

    Pairing file is a CSV containing (ref_genome, gene_pred_path, fasta_file).

    This method handles data loading only. Transcripts are associated in
    the `add_ortholog_to_transcript_dict` method.

    Args:
        pair_file_path: Path to pair file generated by Zoonomia Downloader.

    Returns:
        List of OrthologData objects, which contain the reference genome,
        orthologous gene preds and assemblies.
    """
    orthologs_data = []

    with open(pair_file_path, "r") as f:
        # Skip header
        f.readline()

        for line in f.readlines():
            ref_genome, annotation, assembly = line.split(", ")
            species_name = annotation.split(ref_genome)[-1].split("_")[1:-1]
            species_binomen = "_".join(species_name)

            print("Loading genome for: ", species_binomen, flush=True)

            refseq_data = prepare_refseq_dataset(
                refseq_path=annotation.strip(),
                fasta_path=assembly.strip(),
                mini_dataset=mini,
                drop_non_nm=False,
                use_human_chrs=False,
                transcript_length_drop=transcript_length_drop,
            )

            orthologs_data.append(
                OrthologData(
                    refseqdataset=refseq_data,
                    species_name=species_binomen,
                    species_pair=ref_genome
                )
            )

    return orthologs_data


def construct_transcript_dict_from_gene_dict(
    gene_dict: dict[str: list],
    pair_file_path: str | None,
    mini: bool,
) -> dict[any, list]:
    """Convert gene dict to transcript dict and inject orthology data.

    1. Takes a gene dictionary (gene_id: [transcripts]) and restructures it to
       a transcript dictionary (transcript: [transcripts])
    2. Injects orthology data into the transcript dictionary by using a
       transcript_id: transcript_object map for pair species

    Args:
        gene_dict: Gene name to transcript mapping.
        pair_file_paths: Path to file of orthologous annotations / assemblies.
        mini: Loads a subset of annotations.

    Returns:
        Transcript - Transcript mapping of transcripts.
    """
    transcript_dict = restructure_dict(gene_dict)

    if pair_file_path is None:
        return transcript_dict

    assert not mini, "Mini dataset not supported for orthologs"
    orthology_refseq_datasets = load_orthologs_by_path(pair_file_path)

    versionless_transcript_id_map = {
        x.transcript_id.split(".")[0]: x for x in transcript_dict.keys()
    }

    for ortholog_data in orthology_refseq_datasets:
        print("Injecting ortholog data for {} - {}".format(
            ortholog_data.species_name,
            ortholog_data.species_pair
        ), flush=True)

        transcript_dict = add_ortholog_to_transcript_dict(
            t_id_to_t=versionless_transcript_id_map,
            ortholog_t_list=ortholog_data.refseqdataset.transcripts,
            t_to_t_dict=transcript_dict,
        )

    return transcript_dict


def train_test_split_homologous(genes, df_homology, test_size=0.2, random_state=None):
    """Split genes into train and test sets such that homologous genes are in the same set

    Args:
        genes (list): List of gene names (strings)
        df_homology (pd.DataFrame): DataFrame with columns 'gene_name' and 'gene_group'
        test_size (float, optional): Defaults to 0.2.
        random_state (int, optional): Defaults to None.

    Returns:
        dict: Dictionary with keys 'train_indices' and 'test_indices'
        containing the indices of the genes in the train and test sets
        respectively
    """
    # Map genes to their respective homology groups
    homology_group_map = df_homology.set_index('gene_name')['gene_group'].to_dict()

    # create a list of homology groups
    gene_groups = list()
    for gene in genes:
        if gene in homology_group_map:
            gene_groups.append(homology_group_map[gene])
        else:
            gene_groups.append(None)
    gene_groups = np.array(gene_groups)
    gene_index = np.arange(len(genes))
    # create a dict of gene group to indexes in the gene list
    group_to_index = dict()
    for i, group in enumerate(gene_groups):
        # if a given gene doesn't have a group don't add it
        if group is None:
            continue
        # if the group is not in the dict add it
        if group not in group_to_index:
            group_to_index[group] = []
        # add the index to the group
        group_to_index[group].append(i)

    np.random.seed(random_state)
    np.random.shuffle(gene_index)
    len_of_train = int(len(gene_index) * (1 - test_size))
    num_samples_in_train = 0

    train_indices = []
    test_indices = []
    seen_groups = set()
    for index in gene_index:
        current_group = gene_groups[index]
        if current_group is not None and current_group in seen_groups:
            continue
        seen_groups.add(current_group)

        if num_samples_in_train < len_of_train:
            if current_group is None:
                train_indices.append(index)
                num_samples_in_train += 1
            else:
                group_indexes = group_to_index[current_group]
                train_indices.extend(group_indexes)
                num_samples_in_train += len(group_indexes)
        else:
            if current_group is None:
                test_indices.append(index)
            else:
                group_indexes = group_to_index[current_group]
                test_indices.extend(group_indexes)

    return {
        'train_indices': train_indices,
        'test_indices': test_indices
    }


def load_homology_df(
    species_name: str,
    homologene_path: str = (
        "/data1/morrisq/ian/rna_contrast/homology_maps_homologene"
    ),
):
    homologene_filename = f"{species_name}_homology_map.csv"
    hom_df = pd.read_csv(f"{homologene_path}/{homologene_filename}")
    return hom_df


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
