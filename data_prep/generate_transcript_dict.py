from __future__ import annotations

import argparse
import pickle
import os

import pandas as pd
from tqdm import tqdm

from data_utils import get_transcript_path, ortho_name_fn, ref_name_fn


class Gene:
    """Data class which associates all transcripts related to a gene."""

    def __init__(self, name):
        self.name = name
        self.splice_isoforms = set()
        self.orthologs = set()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def add_transcript_splice(self, transcript_id):
        self.splice_isoforms.update([transcript_id])

    def add_transcript_ortholog(self, transcript_id):
        self.orthologs.update([transcript_id])

    def get_transcript_map(self, add_ortho: bool) -> dict[str, set[str]]:
        """Get dict that associates transcripts in gene with each other.

        All splice isoforms are associated with other splice isoforms. Ortholog
        relationships are optionally added.

        Args:
            add_orthro: Determines whether orthologous relationships are added.

        Returns:
            Dictionary of each transcript and list of associated transcripts.
        """
        tt_dict = {}

        if len(self.splice_isoforms) == 0 and len(self.orthologs) > 0:
            for ortholog_id in self.orthologs:
                tt_dict[ortholog_id.split("_")[0]] = set()
        else:
            for transcript in self.splice_isoforms:
                tt_dict[transcript] = set()
                tt_dict[transcript].update(self.splice_isoforms - {transcript})

        if add_ortho:
            for transcript in tt_dict.keys():
                tt_dict[transcript].update(self.orthologs)

        return tt_dict

    def merge(self, other: Gene):
        """Merge transcripts associated with other Gene into current Gene."""
        self.splice_isoforms.update(other.splice_isoforms)
        self.orthologs.update(other.orthologs)


class TranscriptGeneDictGenerator:
    """Constructs dictionary of genes from genomic annotation files."""

    def __init__(self, save_root_dir: str, source_type: str):
        """Initialize TranscriptGeneDictGenerator.

        Args:
            save_root_dir: Path to directory where transcripts are stored.
            source_type: Type of relationships contained in annotation.
        """
        self.save_root_dir = save_root_dir

        if source_type not in ["REF", "ORTHO"]:
            raise ValueError("Invalid source type.")

        self.source_type = source_type

    def check_transcript_exists(self, transcript_id: str) -> bool:
        """Check that transcript with given id exists.

        Args:
            transcript_id: ID of transcript to check.

        Returns:
            Whether transcript exists at the save directory.
        """
        t_path = "{}/{}/{}.npz".format(
            self.save_root_dir,
            get_transcript_path(transcript_id),
            transcript_id
        )

        return os.path.exists(t_path)

    def parse_splice(self, gp_path: str) -> dict[str, Gene]:
        """Parses gene pred annotation file and loads genes.

        Each transcript is stored by gene name.

        Args:
            gp_path: Path to the gene pred file.

        Returns:
            A mapping from gene name to its object representation.
        """
        anno_df = pd.read_csv(gp_path, sep="\t")

        gene_dict: dict[str, Gene] = {}

        for _, row in anno_df.iterrows():
            transcript_id, gene_name = ref_name_fn(None, row)

            if not self.check_transcript_exists(transcript_id):
                continue

            if gene_name not in gene_dict:
                gene_dict[gene_name] = Gene(gene_name)

            gene_dict[gene_name].add_transcript_splice(transcript_id)

        return gene_dict

    def parse_ortholog(self, gp_path: str, fa_path: str) -> dict[str, Gene]:
        """Generate transcript-transcript dictionary for single species.

        Args:
            gp_path: Path to gene pred file for species.
            fa_path: Path to assembly fasta.

        Returns:
            Map of transcript in reference species to orthologs.
        """
        gp_df = pd.read_csv(gp_path, sep="\t")

        gene_map: dict[str, Gene] = {}

        for _, row in gp_df.iterrows():
            transcript_id, gene_name = ortho_name_fn(fa_path, row)

            if not self.check_transcript_exists(transcript_id):
                continue

            if gene_name not in gene_map:
                gene_map[gene_name] = Gene(gene_name)

            gene_map[gene_name].add_transcript_ortholog(transcript_id)

        return gene_map

    def process_batch(self, batch_file_path: str) -> dict[str, Gene]:
        """Parse all annotations in a batch file into Gene representations.

        Batch files are CSVs with format:

            (ref_species, gene_pred_path, assembly_path)

        Args:
            batch_file_path: Path to batch file.
        """
        batch_df = pd.read_csv(batch_file_path, sep=", ")

        gene_map: dict[str, Gene] = {}

        for _, species_row in tqdm(batch_df.iterrows(), total=len(batch_df)):
            if self.source_type == "REF":
                species_dict = self.parse_splice(
                    species_row.iloc[1],
                )
            elif self.source_type == "ORTHO":
                species_dict = self.parse_ortholog(
                    species_row.iloc[1], species_row.iloc[2]
                )
            else:
                raise ValueError("Unknown source type.")

            for gene_name, gene in species_dict.items():
                if gene_name in gene_map:
                    gene_map[gene_name].merge(gene)
                else:
                    gene_map[gene_name] = gene

        return gene_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_save_root_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--source_type", type=str)
    parser.add_argument("--batch_path", type=str)
    args = parser.parse_args()

    out_path = "{}/{}-{}.pkl".format(
        args.output_dir,
        args.source_type,
        args.output_name,
    )

    mapper = TranscriptGeneDictGenerator(
        args.transcript_save_root_dir,
        args.source_type
    )

    tt_map = mapper.process_batch(args.batch_path)

    with open(out_path, "wb") as out_f:
        pickle.dump(tt_map, out_f, protocol=pickle.HIGHEST_PROTOCOL)
