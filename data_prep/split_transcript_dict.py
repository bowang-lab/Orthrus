import argparse
import concurrent.futures
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from data_utils import get_transcript_path
from generate_transcript_dict import Gene


class DatasetSaver:
    """Data class which constructs map of transcript to related transcripts.

    Transcripts associated in this map are used as the positive pair
    relationships during contrastive pre-training.

    This class also can also generate maps which filter out transcripts above
    a certain length. This is useful for optimizing GPU memory usage in multi-
    GPU training, where longer transcripts can be sent to a single node to
    reduce length padding.
    """

    def __init__(self, npz_save_dir: str, split_length: int, max_length: int):
        """Initialize a transcript dataset.

        Args:
            npz_save_dir: Directory where transcripts are saved.
            split_length: Transcripts above and below this length can be
                optionally partitioned into two different maps.
            max_length: Transcripts above this length are filtered.
        """
        self.npz_save_dir = npz_save_dir
        self.split_length = split_length
        self.max_length = max_length

        self.transcript_map: dict[str, set[str]] = None
        self.transcript_length_map: dict[str:int] = {}

    def load_from_tt_maps(self, paths: list[str]):
        out_map: dict[str, set[str]] = {}

        for path in tqdm(paths):
            with open(path, "rb") as f:
                t_map = pickle.load(f)

                for t_id, t_pairs in t_map.items():
                    if t_id in out_map:
                        out_map[t_id].update(set(t_pairs))
                    else:
                        out_map[t_id] = set(t_pairs)

        self.transcript_map = out_map

    def load_from_gt_maps(self, paths: list[str], homo_path: str = None):
        overall_gene_map: dict[str, Gene] = {}

        for path in tqdm(paths):
            with open(path, "rb") as f:
                gene_map = pickle.load(f)

                for gene_name, gene in gene_map.items():
                    if gene_name in overall_gene_map:
                        overall_gene_map[gene_name].merge(gene)
                    else:
                        overall_gene_map[gene_name] = gene

        if homo_path is not None:
            overall_gene_map = self.map_homology(overall_gene_map, homo_path)

        tt_map: dict[str, set[str]] = {}

        for gene in tqdm(overall_gene_map.values()):
            gene_tt_map = gene.get_transcript_map(add_ortho=True)

            for transcript, pair_set in gene_tt_map.items():
                if transcript in tt_map:
                    tt_map[transcript].update(pair_set)
                else:
                    tt_map[transcript] = pair_set

        self.transcript_map = tt_map

    def map_homology(self, gt_map: dict[str, Gene], homology_map_path: str):
        homology_map = pd.read_csv(homology_map_path, sep=",")

        gene_set_map: dict[int, set[str]] = {}

        for i, row in homology_map.iterrows():
            gene_name = row.iloc[1].lower()
            gene_group = row.iloc[2]

            if gene_group not in gene_set_map:
                gene_set_map[gene_group] = set([gene_name])
            else:
                gene_set_map[gene_group].add(gene_name)

        gene_gene_map = {}

        for _, gene_set in gene_set_map.items():
            for gene in list(gene_set):
                gene_gene_map[gene] = gene_set - set([gene])

        for gene in gt_map.keys():
            if gene in gene_gene_map:
                homo_genes = gene_gene_map[gene]

                for homo_gene in homo_genes:
                    if homo_gene in gt_map:
                        gt_map[gene].merge(gt_map[homo_gene])

        return gt_map

    def get_transcript_length(self, transcript_id: str) -> int:
        """Get length of transcript.

        Args:
            transcript_id: Transcript id.

        Returns:
            Length of transcript in nucleotides. Returns -1 if not found.
        """
        if transcript_id in self.transcript_length_map:
            return self.transcript_length_map[transcript_id]

        t_path_dirs = get_transcript_path(transcript_id)
        t_path = "{}/{}/{}.npz".format(
            self.npz_save_dir,
            t_path_dirs, transcript_id
        )

        try:
            t_length = int(np.load(t_path)["length"])
        except FileNotFoundError:
            t_length = -1

        self.transcript_length_map[transcript_id] = t_length
        return t_length

    def filter_by_length(self, p_ids: list[str], is_short: bool) -> list[str]:
        """Filters all transcript in list by length threshold.

        Method is I/O bound due to reading transcript files, and is
        parallelized using threads for efficiency.

        Args:
            p_ids: List of transcripts to filter.
            is_short: If true, selects transcripts below self.split_length.
                Otherwise, selects transcripts less than self.max_length.

        Returns:
            Filtered list of transcripts.
        """
        e = concurrent.futures.ThreadPoolExecutor(max_workers=len(p_ids))

        futures = {}
        for p_id in p_ids:
            future = e.submit(self.get_transcript_length, p_id)
            futures[future] = p_id

        new_associates = []
        for future in concurrent.futures.as_completed(futures):
            p_id = futures[future]

            p_length = future.result()

            if p_length == -1:
                continue

            if is_short and p_length < self.split_length:
                new_associates.append(p_id)
            elif not is_short and p_length < self.max_length:
                new_associates.append(p_id)

        e.shutdown(wait=True)
        return new_associates

    def split_transcript_map_by_length(
        self,
        transcript_map: dict[str: list[str]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Filter transcript map by length of transcript.

        Returns:
            Transcript map with elements having length within bounds.
        """
        shorter_map = {}
        longer_map = {}

        for t_id in tqdm(transcript_map.keys()):
            t_len = self.get_transcript_length(t_id)

            if t_len >= self.max_length or t_len == -1:
                continue

            is_short = t_len < self.split_length

            pair_ts = transcript_map[t_id]

            if len(pair_ts) > 0:
                new_associates = self.filter_by_length(pair_ts, is_short)
            else:
                new_associates = []

            if len(new_associates) > 0:
                if not is_short:
                    longer_map[t_id] = new_associates
                else:
                    shorter_map[t_id] = new_associates

        return shorter_map, longer_map

    def split_transcript_map_by_length_chunked(
        self,
        chunk: int,
        total_chunks: int
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Perform length filtering of transcripts on dataset chunk.

        Args:
            chunk: Current chunk to process.
            total_chunks: Number of total chunks dataset is divided into.

        Returns:
            Filtered transcript maps for current chunk.
        """
        assert self.transcript_map is not None

        keys = list(self.transcript_map.keys())
        chunk_keys = np.array_split(keys, total_chunks)[chunk]

        chunk_transcript_map = {}

        for k in chunk_keys:
            chunk_transcript_map[k] = list(self.transcript_map[k])

        return self.split_transcript_map_by_length(chunk_transcript_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_combo", type=str)
    parser.add_argument("--current_chunk", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--split_length", type=int, default=3900)
    parser.add_argument("--max_length", type=int, default=12288)
    parser.add_argument("--source_type", type=str, default="GENE")
    parser.add_argument("--homo_path", type=str)
    args = parser.parse_args()

    MAP_ROOT = ""
    TRANSCRIPT_ROOT = ""

    def prepend_path(map_list):
        return [MAP_ROOT + "/" + map_name for map_name in map_list]

    dataset_combos = {
        "ORTHO-eutheria": [
            "ORTHO-human_eutheria.pkl",
            "ORTHO-mouse_eutheria.pkl"
        ],
        "REF-splice_all_basic": ["REF-splice_all_basic.pkl"],
        "REF-splice_two_basic": ["REF-splice_two_basic.pkl"],
        "COMB-eutheria-splice_all_basic": [
            "REF-splice_all_basic.pkl",
            "ORTHO-human_eutheria.pkl",
            "ORTHO-mouse_eutheria.pkl",
        ],
        "COMB-eutheria-splice_two_basic": [
            "REF-splice_two_basic.pkl",
            "ORTHO-human_eutheria.pkl",
            "ORTHO-mouse_eutheria.pkl",
        ],
    }

    transcript_maps = prepend_path(dataset_combos[args.dataset_combo])

    tt_splitter = DatasetSaver(
        npz_save_dir=TRANSCRIPT_ROOT,
        split_length=args.split_length,
        max_length=args.max_length,
    )

    if args.source_type == "GENE":
        tt_splitter.load_from_gt_maps(transcript_maps, args.homo_path)
    elif args.source_type == "TRANSCRIPT":
        tt_splitter.load_from_tt_maps(transcript_maps)
    else:
        raise ValueError("Invalid source type.")

    short_map, long_map = tt_splitter.split_transcript_map_by_length_chunked(
        args.current_chunk, args.total_chunks
    )

    out_dir = Path(MAP_ROOT) / "splits"
    out_dir.mkdir(exist_ok=True, parents=True)

    if args.homo_path is not None:
        args.dataset_combo += "-homo"

    short_fn = "{}/{}_shorter_{}.pkl".format(
        out_dir, args.dataset_combo, args.current_chunk
    )

    long_fn = "{}/{}_longer_{}.pkl".format(
        out_dir, args.dataset_combo, args.current_chunk
    )

    with open(short_fn, "wb") as f:
        pickle.dump(short_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(long_fn, "wb") as f:
        pickle.dump(long_map, f, protocol=pickle.HIGHEST_PROTOCOL)
