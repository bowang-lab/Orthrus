import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from orthrus.dilated_resnet import dilated_small, not_dilated_small
from orthrus.mamba import MixerModel
from orthrus.util import train_test_split_homologous, load_homology_df
from orthrus.eval_utils import get_representations


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_model(
    repository: str, 
    run_name: str, 
    model_name: str,
    n_tracks: int = None,
    load_state_dict: bool = True,
) -> nn.Module:
    """Load trained model located at specified path."""
    # check if model_config.json exists in the repository
    config_path = os.path.join(repository, run_name, "model_config.json")
    
    if not os.path.exists(config_path):
        raise RuntimeError(f"Model config not found. {repository}/{run_name}")
    else:
        with open(config_path, "r") as f:
            model_params = json.load(f)
            # del model_params['temperature']
    config_path = os.path.join(repository, run_name, "data_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Data config not found. {repository}/{run_name}")
    else:
        with open(config_path, "r") as f:
            data_params = json.load(f)
            # del model_params['temperature']

    # list all the checkpoints in run_name directory and find the 
    # the lastest one of the format epoch=18-step=20000.ckpt
    if model_name == 'latest':
        checkpoints = os.listdir(os.path.join(repository, run_name))
        checkpoints = [f for f in checkpoints if f.endswith('.ckpt')]
        model_name = max(checkpoints, key=lambda x: int(x.split('=')[2].rstrip('.ckpt').rstrip('-v1')))
        
    model_path = os.path.join(repository, run_name, model_name)

    if model_params["model_class"] == "resnet":
        del model_params["model_class"]
        del model_params["temperature"]
        if model_params["resnet"] == "not_dilated_small":
            model = not_dilated_small(**model_params)
        elif model_params["resnet"] == "dilated_small":
            del model_params["resnet"]
            model = dilated_small(**model_params)
        else:
            raise ValueError("Model config missing resnet type.")
    elif model_params['model_class'] == "ssm":
        del model_params["model_class"]
        if n_tracks is None:
            n_tracks = data_params['n_tracks']
        model = MixerModel(
            d_model=model_params["ssm_model_dim"],
            n_layer=model_params["ssm_n_layers"],
            input_dim=n_tracks
        )
    else:
        raise ValueError("Unknown model type")

    checkpoint = torch.load(model_path)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict[k.lstrip("model")[1:]] = v

    if load_state_dict:
        model.load_state_dict(state_dict)

    return model

def plot_performance_over_epochs(df, model_name, plot_save_dir):
    sns.set(style="whitegrid")
    sns.set_context('talk')
    sns.set_palette("colorblind")
    plt.figure(figsize=(12, 8))

    # Create a seaborn lineplot
    sns.lineplot(
        x='step', 
        y='mean_value', 
        hue='key', 
        style='key', 
        markers=True, 
        dashes=False, 
        data=df[
            # (~df['key'].str.contains('loss')) & 
            (~df['key'].str.contains('roc'))
        ]
    )
    # Add title and labels
    plt.title(f'Performance Metrics over training for Different Datasets {model_name}')
    plt.xlabel('step')
    # move legend outside plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Mean Value')
    # plot text for maximum point for each line
    name_to_index = df[
        (~df['key'].str.contains('loss')) & 
        (~df['key'].str.contains('roc'))
    ].groupby('key').agg({'mean_value': 'idxmax'}).to_dict()['mean_value']
    name_to_index2 = df[
        df['key'].str.contains('loss')
    ].groupby('key').agg({'mean_value': 'idxmin'}).to_dict()['mean_value']
    name_to_index.update(name_to_index2)

    for name, index in name_to_index.items():
        plt.text(
            df.loc[index]['step'], 
            df.loc[index]['mean_value'] + 0.01, 
            f"{df.loc[index]['mean_value']:.2f} {name}", 
            fontsize=12
        )

    # Add a text in bottom right summarizing mean_value at last epoch
    last_epoch_vals = df[
        df['step'] == df['step'].max()
    ][['mean_value','key']]
    last_epoch_vals = {x: round(y,2) for x, y in zip(last_epoch_vals['key'], last_epoch_vals['mean_value'])}

    last_epoch_vals = pd.DataFrame.from_dict(last_epoch_vals, orient='index', columns=['mean_value'])
    plt.text(
        1.25, 0.05, 
        f"Mean Value at Last Step: {last_epoch_vals.to_string()}",
        fontsize=12, 
        ha='center', 
        va='center', 
        transform=plt.gca().transAxes
    )
    plt.ylim(0.3, 0.75)
    # Display the plot
    plt.savefig(f'{plot_save_dir}/all_{model_name}_performance.png')
    plt.show()

def perform_linear_eval(
    dataset_name: str,
    dataset_config: dict,
    embedder_model: nn.Module,
    args: dict
) -> list[dict]:
    ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    performance_list = list()
    # do pretty print of dataset_name
    if args['verbose'] > 0:
        print('---------')
        print(dataset_name)
        print('---------')

    args['lin_model'] = dataset_config['linear_model']
    args['emb_dir'] = os.path.join(
        args['npz_dir'],
        dataset_config['file_name']
    )

    data = np.load(args['emb_dir'])

    if args['n_tracks'] == 4:
        dataset = data['X'][:, :, :4]
    else:
        dataset = data['X']

    embs = get_representations(
        model=embedder_model,
        seq_array=dataset.astype(np.float32),
        batch_size=args["batch_size"],
        channel_last=True
    )
    
    if args['lin_model'] != "logistic":
        labs = data["y"].flatten()
    else:
        labs = data["y"]

    np.random.seed(args['split_random_seed'])
    if not args['homology_split']:
        train_X, vt_X, train_y, vt_y = train_test_split(embs, labs, test_size=0.3)
        val_X, test_X, val_y, test_y = train_test_split(vt_X, vt_y, test_size=0.5)
        if args['verbose'] > 1:
            print('Random split:')
            print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
    else:
        gene_names = data['genes']
        homology_df = load_homology_df(dataset_config['species'])
        
        split = train_test_split_homologous(
            gene_names, homology_df, test_size=0.3, random_state=args['split_random_seed']
        )
        train_X = embs[split['train_indices']]
        train_y = labs[split['train_indices']]
        val_X = embs[split['test_indices']]
        val_y = labs[split['test_indices']]
        val_genes = gene_names[split['test_indices']]
        
        # Second split
        split = train_test_split_homologous(
            val_genes, homology_df, test_size=0.5, random_state=args['split_random_seed']
        )
        test_X = val_X[split['test_indices']]
        test_y = val_y[split['test_indices']]
        val_X = val_X[split['train_indices']]
        val_y = val_y[split['train_indices']]
        val_genes = val_genes[split['train_indices']]
        if args['verbose'] > 1:
            print('Homology split:')
            print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

    # If data_fraction is less than 1.0, subsample the data
    if args['data_fraction'] < 1.0:
        train_X, _, train_y, _ = train_test_split(
            train_X, train_y, train_size=args['data_fraction'], random_state=args['split_random_seed']
        )
    
    train_X = train_X.reshape(train_X.shape[0], -1)
    val_X = val_X.reshape(val_X.shape[0], -1)

    if args["lin_model"] == "linear":
        model = LinearRegression().fit(train_X, train_y)
    elif args["lin_model"] == "ridge":
        model = RidgeCV(alphas=ALPHAS).fit(train_X, train_y)
    elif args["lin_model"] == "logistic":
        model = MultiOutputClassifier(
            LogisticRegression(max_iter=5000)
        ).fit(train_X, train_y)
    else:
        raise ValueError("Unknown linear model.")

    if args['lin_model'] != "logistic":
        train_pred = model.predict(train_X)
        val_pred = model.predict(val_X)
        test_pred = model.predict(test_X)

        t_loss = np.mean((train_y - train_pred) ** 2)
        v_loss = np.mean((val_y - val_pred) ** 2)
        test_loss = np.mean((test_y - test_pred) ** 2)

        t_r = pearsonr(train_pred, train_y).statistic
        v_r = pearsonr(val_pred, val_y).statistic
        test_r = pearsonr(test_pred, test_y).statistic

        train_metrics = "Train Loss: {:.4f}, Train R: {:.4f},"
        val_metrics = "Val Loss: {:.4f}, Val R: {:.4f}"
        test_metrics = "Test Loss: {:.4f}, Test R: {:.4f}"

        if not args["full_eval"]:
            metrics = [v_loss, v_r]
            labels = ["val_loss", "val_r"]
            splits = ["val", "val"]
        else:
            metrics = [
                t_loss,
                v_loss,
                test_loss,
                t_r,
                v_r,
                test_r
            ]
            labels = [
                "train_loss",
                "val_loss",
                "test_loss",
                "train_r",
                "val_r",
                "test_r"
            ]
            splits = ["train", "val", "test"] * 2

        for metric in zip(metrics, labels, splits):
            performance_list.append({
                "dataset": dataset_name,
                "run_name": args["run_name"],
                "model_name": args["model_name"],
                "lin_model": args["lin_model"],
                "metric": metric[1],
                "value": metric[0],
                "split": metric[2]
            })

        train_metrics = train_metrics.format(t_loss, t_r)
        val_metrics = val_metrics.format(v_loss, v_r)
        test_metrics = test_metrics.format(test_loss, test_r)
    else:
        n_class = len(data["y"][0])

        train_pred = flatten(model.predict_proba(train_X), n_class)
        val_pred = flatten(model.predict_proba(val_X), n_class)
        test_pred = flatten(model.predict_proba(test_X), n_class)

        t_roc_auc = roc_auc_score(train_y.flatten(), train_pred)
        v_roc_auc = roc_auc_score(val_y.flatten(), val_pred)
        test_roc_auc = roc_auc_score(test_y.flatten(), test_pred)

        t_auprc = average_precision_score(train_y.flatten(), train_pred)
        v_auprc = average_precision_score(val_y.flatten(), val_pred)
        test_auprc = average_precision_score(test_y.flatten(), test_pred)

        if not args["full_eval"]:
            metrics = [v_roc_auc, v_auprc]
            labels = ["val_roc_auc", "val_auprc"]
            splits = ["val", "val"]
        else:
            metrics = [
                t_roc_auc,
                v_roc_auc,
                test_roc_auc,
                t_auprc,
                v_auprc,
                test_auprc
            ]
            labels = [
                "train_roc_auc",
                "val_roc_auc",
                "test_roc_auc",
                "train_auprc",
                "val_auprc",
                "test_auprc"
            ]
            splits = ["train", "val", "test"] * 2

        for metric in zip(metrics, labels, splits):
            performance_list.append({
                "dataset": dataset_name,
                "run_name": args["run_name"],
                "model_name": args["model_name"],
                "lin_model": args["lin_model"],
                "metric": metric[1],
                "value": metric[0],
                "split": metric[2]
            })

        train_metrics = "Train AUROC: {:.4f}, Train AUPRC: {:.4f}"
        val_metrics = "Val AUROC: {:.4f}, Val AUPRC: {:.4f}"
        test_metrics = "Test AUROC: {:.4f}, Test AUPRC: {:.4f}"

        train_metrics = train_metrics.format(t_roc_auc, t_auprc)
        val_metrics = val_metrics.format(v_roc_auc, v_auprc)
        test_metrics = test_metrics.format(test_roc_auc, test_auprc)

    # Print results
    info = "Model: {}, Regressor: {}, Task: {}".format(
        args["model_name"],
        args["lin_model"],
        dataset_name
    )

    if args["verbose"] > 1:
        if not args["eval_test"]:
            print(info)
            print(train_metrics)
            print(val_metrics)
        else:
            print(info)
            print(test_metrics)

    return performance_list


def perform_eval(
    embedder_model,
    args,
    datasets
):
    performance_list = list()

    for dataset_name, dataset_config in datasets.items():
        for i in range(args["n_seeds"]):
            args["split_random_seed"] = args["split_random_seed"] + i
            one_data_performances = perform_linear_eval(
                dataset_name,
                dataset_config,
                embedder_model,
                args,
            )
            performance_list.extend(one_data_performances)

    # Save results
    performance_df = pd.DataFrame(performance_list)

    group_cols = ["dataset", "metric", "split"]

    mean_vals = performance_df[group_cols + ["value"]].groupby(group_cols)
    mean_vals = mean_vals.mean().reset_index()
    mean_vals.rename(columns={'value': 'mean_value'}, inplace=True)

    std_vals = performance_df[group_cols + ["value"]].groupby(group_cols)
    std_vals = std_vals.std().reset_index()
    std_vals.rename(columns={'value': 'std_value'}, inplace=True)

    performance_df.drop(columns=['value'], inplace=True)
    performance_df = performance_df[~performance_df.duplicated()]

    performance_df = performance_df.merge(mean_vals, on=group_cols)
    performance_df = performance_df.merge(std_vals, on=group_cols)

    if not os.path.exists(args['results_dir']):
        os.makedirs(args['results_dir'])
    performance_df['key'] = performance_df['dataset'] + '_' + performance_df['metric']

    performance_df.to_csv(
        os.path.join(
            args['results_dir'],
            f"linear_probe_results_{args['run_name']}_{args['model_name']}_{args['data_fraction']}.csv"
        ),
        index=False
    )

    if args['verbose'] > 0:
        print(performance_df['run_name'].iloc[0])
        print(performance_df['model_name'].iloc[0])

        print(
            performance_df[
                performance_df['split'] == 'val'
            ][['dataset', 'metric', 'mean_value']].to_csv(index=False)
        )

    return performance_df


def eval_all_checkpoints(args, datasets):
    # identify all model names with the structure epoch=7-step=8000.ckpt
    files_in_dir = os.listdir(os.path.join(args["model_repository"], args["run_name"]))
    model_names = [f for f in files_in_dir if f.endswith('.ckpt')]
    performances = {}
    print(f"Found {len(model_names)} models to evaluate.")
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        embedder_model = load_model(
            repository=args["model_repository"],
            run_name=args["run_name"],
            model_name=model_name,
            n_tracks=args['n_tracks'],
            load_state_dict=args['load_state_dict']
        ).cuda()
        performance_df = perform_eval(embedder_model, args, datasets)
        performances[model_name] = performance_df
        
    return performances
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str)
    parser.add_argument("--model_repository", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--split_random_seed", type=int, default=2547)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--eval_test", type=bool, default=False)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default='./results')
    parser.add_argument("--full_eval", action="store_true")
    parser.add_argument("--n_tracks", type=int, default=6)
    parser.add_argument("--load_state_dict", type=str2bool, default='True')
    parser.add_argument("--homology_split", type=str2bool, default='True')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_fraction", type=float, default=1.0)

    args = vars(parser.parse_args())
    if args['verbose'] > 1:
        for key, value in args.items():
            print('Config:')
            print(f"{key}: {value}")

    datasets = {
        'go_mf': {
            'file_name': "go_dna_dataset.npz",
            'linear_model': "logistic",
            'species': 'human',
        },
        'mrfp': {
            'file_name': "mrfp.npz",
            'linear_model': "ridge",
            'species': 'human',
        },
        'mrl': {
            'file_name': "mrl_isoform_resolved.npz",
            'linear_model': "ridge",
            'species': 'human',            
        },
        'protein_loc': {
            'file_name': "protein_localization_dataset.npz",
            'linear_model': "logistic",
            'species': 'human',
        },
        'hl_human': {
            'file_name': "rna_hl_human.npz",
            'linear_model': "ridge",
            'species': 'human'
        },
        'hl_mouse': {
            'file_name': "rna_hl_mouse.npz",
            'linear_model': "ridge",
            'species': 'mouse',
        },
    }

    # perform all evaluations
    if args['model_name'] == "all":
        performances = eval_all_checkpoints(args, datasets)
        dfs = []
        for model_name, df in performances.items():
            df['model_name'] = model_name
            dfs.append(df)
            df['epoch'] = df['model_name'].apply(lambda x: int(x.split('=')[1].rstrip('-step')))
            df['step'] = df['model_name'].apply(lambda x: int(x.split('=')[2].rstrip('.ckpt')))

        df = pd.concat(dfs)
        df.to_csv(
            os.path.join(
                args['results_dir'],
                f"linear_probe_results_{args['run_name']}_all.csv"
            ),
            index=False
        )
        plot_performance_over_epochs(df, args['run_name'], args['results_dir'])
        return 
    
    
    embedder_model = load_model(
        repository=args["model_repository"],
        run_name=args["run_name"],
        model_name=args["model_name"],
        n_tracks=args['n_tracks'],
        load_state_dict=args['load_state_dict'],
    ).cuda()

    _ = perform_eval(
        embedder_model,
        args,
        datasets
    )


def auprc_mc(true, pred, n_class):
    average_precision_per_class = []
    for i in range(n_class):
        class_lab = (true == i).astype(int)
        class_prob = pred[:, i]

        class_auprc = average_precision_score(class_lab, class_prob)

        average_precision_per_class.append(class_auprc)
    return np.mean(average_precision_per_class)


def flatten(arr, n_class):
    """Flatten multilabel probability output. Takes probability of positive."""
    out = []
    for n in range(arr[0].shape[0]):
        for c in range(n_class):
            out.append(arr[c][n][1])
    return np.array(out)


if __name__ == "__main__":
    main()
