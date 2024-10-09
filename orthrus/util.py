import pandas as pd
from datetime import datetime
import re
from absl import flags
import numpy as np
from torchmetrics import Metric
import torch
from torchmetrics.utilities import dim_zero_cat


def make_timestamp():
    timestamp = "_".join(re.split(":|-| ", str(datetime.now()).split(".")[0]))
    return timestamp


def load_appris(unique_transcripts=True):
    # generate doc string
    """
    Load the appris data
    :param unique_transcripts: whether to load only unique transcripts
    :return: the appris data
    """
    # ## load human appris
    dir = '/h/phil/Documents/01_projects/contrastive_rna_representation/'

    app_h = pd.read_csv(f'{dir}/data/appris_data_human.principal.txt', sep='\t')
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



def pretty_print_flags():
    FLAGS = flags.FLAGS
    print("Configuration:")
    for flag_name in sorted(FLAGS):
        flag_value = FLAGS[flag_name].value
        print(f"{flag_name}: {flag_value}")


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
        '/scratch/hdd001/home/phil/rna_contrast/datasets/'
        'data_for_ian/annotation_data/homology_maps_homologene'
    ),
):
    homologene_filename = f"{species_name}_homology_map.csv"
    hom_df = pd.read_csv(f"{homologene_path}/{homologene_filename}")
    return hom_df


class PearsonR(Metric):
    def __init__(self, num_targets=1, summarize=True):
        super().__init__()
        self.num_targets = num_targets
        self.summarize = summarize
        self.add_state("count", default=torch.zeros(num_targets), dist_reduce_fx="sum")
        self.add_state("product", default=torch.zeros(num_targets), dist_reduce_fx="sum")
        self.add_state("true_sum", default=torch.zeros(num_targets), dist_reduce_fx="sum")
        self.add_state("true_sumsq", default=torch.zeros(num_targets), dist_reduce_fx="sum")
        self.add_state("pred_sum", default=torch.zeros(num_targets), dist_reduce_fx="sum")
        self.add_state("pred_sumsq", default=torch.zeros(num_targets), dist_reduce_fx="sum")

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.squeeze().float()
        y_pred = y_pred.squeeze().float()

        reduce_axes = 0 if len(y_true.shape) == 1 else [0]

        product = torch.sum(y_true * y_pred, dim=reduce_axes)
        true_sum = torch.sum(y_true, dim=reduce_axes)
        true_sumsq = torch.sum(y_true ** 2, dim=reduce_axes)
        pred_sum = torch.sum(y_pred, dim=reduce_axes)
        pred_sumsq = torch.sum(y_pred ** 2, dim=reduce_axes)
        count = torch.sum(torch.ones_like(y_true), dim=reduce_axes)

        self.product += product.unsqueeze(0)
        self.true_sum += true_sum.unsqueeze(0)
        self.true_sumsq += true_sumsq.unsqueeze(0)
        self.pred_sum += pred_sum.unsqueeze(0)
        self.pred_sumsq += pred_sumsq.unsqueeze(0)
        self.count += count.unsqueeze(0)

    def compute(self):
        true_mean = self.true_sum / self.count
        true_mean2 = true_mean ** 2
        pred_mean = self.pred_sum / self.count
        pred_mean2 = pred_mean ** 2

        term1 = self.product
        term2 = -true_mean * self.pred_sum
        term3 = -pred_mean * self.true_sum
        term4 = self.count * true_mean * pred_mean
        covariance = term1 + term2 + term3 + term4

        true_var = self.true_sumsq - self.count * true_mean2
        pred_var = self.pred_sumsq - self.count * pred_mean2
        pred_var = torch.where(pred_var > 1e-12, pred_var, torch.full_like(pred_var, float('inf')))

        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var

        if self.summarize:
            return torch.mean(correlation)
        else:
            return correlation
        

class SpearmanR(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.eps = 1e-8
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds.squeeze())
        self.target.append(target.squeeze())

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        # some intermediate computation...
        r_preds, r_target = self._rank_data(preds), self._rank_data(target)
        preds_diff = r_preds - r_preds.mean(0)
        target_diff = r_target - r_target.mean(0)
        cov = (preds_diff * target_diff).mean(0)
        preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
        target_std = torch.sqrt((target_diff * target_diff).mean(0))
        # finalize the computations
        corrcoef = cov / (preds_std * target_std + self.eps)
        return torch.clamp(corrcoef, -1.0, 1.0)
    
    def _rank_data(self, data: torch.Tensor) -> torch.Tensor:
        return data.argsort().argsort().to(torch.float32)