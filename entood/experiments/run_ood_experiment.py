from analogyproj.experiments.cifar100_coarse_label_split import (
    make_experiment_baseline_no_ae,
    make_experiment_baseline_ft_ae,
    make_experiment_platostore_ft_pae,
    generate_data
)
from analogyproj.analysis.classification import plot_confusion

import argparse
import json
from pathlib import Path
import time

import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import wandb


def plot_distribution_confusions(clf, 
                                 datasets,
                                 save_file=None,
                                 show_fig=False,
                                 class_names=None):
    fig = plt.figure(constrained_layout=True, figsize=(13, 12))
    n = 20
    gs = fig.add_gridspec(n, n+1)
    axs = [[
            fig.add_subplot(gs[i*n//2:(n//2)*(i+1), (n//2)*j:(n//2)*(j+1)]) 
            for j in range(2)
        ] for i in range(2)
    ]
    cbar_ax = fig.add_subplot(gs[0:n, n])

    if not class_names:
        from analogyproj.data_prep.cifar100_utils import read_cifar100_pretty_labels
        _, class_names = read_cifar100_pretty_labels()

    for i, (name, ds) in enumerate(datasets):
        ax = axs[i // 2][i % 2]
        plot_confusion(clf, ds, class_names=class_names, 
                       ax=ax, cbar=i%2, cbar_ax=cbar_ax,
                       x_tick_rotation=90, y_tick_rotation=0, title=name)
        if i < 2:
            ax.set_xticklabels([])
        if i % 2 == 1:
            ax.set_yticklabels([])

    if save_file:
        plt.savefig(save_file)
    
    # TODO: convert to image log rather than plotly log
    wandb.log({'distribution_confusion_plot': wandb.Image(plt)})
    
    if show_fig:
        plt.show()

        
def compute_ood_accs(model, datasets, verbose=0):
    train_ds_ind, test_ds_ind, train_ds_ood, test_ds_ood = datasets
    ind_train_acc = model.evaluate(train_ds_ind, return_dict=True, verbose=0)['acc']
    ind_test_acc = model.evaluate(test_ds_ind, return_dict=True, verbose=0)['acc']
    ood_train_acc = model.evaluate(train_ds_ood, return_dict=True, verbose=0)['acc']
    ood_test_acc = model.evaluate(test_ds_ood, return_dict=True, verbose=0)['acc']
    acc_values = {
        'ind_train_acc': float(ind_train_acc),
        'ind_test_acc': float(ind_test_acc),
        'ood_train_acc': float(ood_train_acc),
        'ood_test_acc': float(ood_test_acc),
    }
    return acc_values


def analyse_ood_experiment(run, datasets):
    train_ds_ind, test_ds_ind, train_ds_ood, test_ds_ood = datasets
    
    print('Computing OOD Accuracy Shift:')
    acc_values = compute_ood_accs(run.model, datasets, verbose=1)
    
    wandb.log(acc_values)
    
    ind_train_acc = acc_values['ind_train_acc']
    ind_test_acc = acc_values['ind_test_acc']
    ood_train_acc = acc_values['ood_train_acc']
    ood_test_acc = acc_values['ood_test_acc']
    print(f'In-distribution train accuracy = {100*ind_train_acc:.2f}%')
    print(f'In-distribution test accuracy = {100*ind_test_acc:.2f}%')
    print(f'Out-distribution accuracy = {100*ood_train_acc:.2f}%')
    
    with open(f'{run.run_dir}/distributional_shift_results.json', mode='w') as f:
        json.dump(acc_values, f)

    datasets_with_labels = [
        ('In-Distribution Training', train_ds_ind), 
        ('In-Distribution Testing', test_ds_ind),
        ('Out-Of-Distribution Training', train_ds_ood), 
        ('Out-Of-Distribution Testing', test_ds_ood),
    ]

    save_file = f'{run.run_dir}/confusion_under_distributional_shift.png'
    plot_distribution_confusions(run.model, datasets_with_labels, save_file=save_file)
    

class AnalyseOODCallback(tf.keras.callbacks.Callback):

    def __init__(self, datasets, monitor='val_acc', point_delta=0.05):
        super(AnalyseOODCallback, self).__init__()
        self.datasets = datasets
        self.monitor = monitor
        self.point_delta = point_delta
        self.last_val = 0
        
    def on_epoch_end(self, epoch, logs):
        if logs[self.monitor] - self.last_val >= self.point_delta:
            acc_values = compute_ood_accs(self.model, self.datasets)      
            logs.update(acc_values)
            self.last_val = logs[self.monitor]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str,
                        help='Name of experiment, one of the following values:'
                             '"baseline_no_ae" (baseline no autoencoder), '
                             '"baseline_ft_ae (baseline fine-tune from pretrained AE), '
                             '"platostore_ft_pae" (baseline fine-tune from pretrained '
                             'AE while also training AE)')
    
    # default hyperparams chosen from sweep on baseline_ft_ae
    parser.add_argument('--learning_rate', type=float, default=0.04,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--optimiser', type=str, default='sgd',
                        help='Type of optimiser to use ("sgd", "adam")')
    parser.add_argument('--momentum', type=float, default=0.45,
                        help='Momentum for an SGD optimiser.')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='Whether or not to use Nesterov momentum for an SGD optimiser.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size to use.')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs to run for.')
    parser.add_argument('--in_dist_prop', type=float, default=0.8,
                        help='Proportion of data to keep for the in-distribution.')
    parser.add_argument('--reduce_lr_on_plateau', type=bool, default=True,
                        help='Whether or not to reduce learning rate on plateau')
    
    
    parser.add_argument('--clf_weight', type=float, default=0.5,
                        help='Weight for classification loss in joint training.')
    parser.add_argument('--ae_weight', type=float, default=0.5,
                        help='Weight for autoencoder loss in joint training.')
    
    return vars(parser.parse_args())


if __name__ == '__main__':
    
    config = parse_args()
    
    datasets, ds_info = generate_data(config)
    train_ds_ind, test_ds_ind, train_ds_ood, test_ds_ood = datasets

    for k, v in ds_info.items():
        print(f'{k}: {v}')
    
    def custom_analyse(run):
        analyse_ood_experiment(run, datasets)
    
    
    experiment = config.pop('experiment')
    if experiment == 'baseline_no_ae':
        make_experiment = make_experiment_baseline_no_ae
    elif experiment == 'baseline_ft_ae':
        make_experiment = make_experiment_baseline_ft_ae
    elif experiment == 'platostore_ft_pae':
        make_experiment = make_experiment_platostore_ft_pae
    else:
        raise ValueError(f'Invalid parameter: {experiment=}')
    
    config['batches_per_epoch'] = ds_info['n_train_ind_batches']
    experiment = make_experiment(train_ds_ind, test_ds_ind, ds_info, config,
                                 custom_analyse=custom_analyse,
                                 callbacks=[AnalyseOODCallback(datasets)])
    
    for k, v in experiment.get_config().items():
        print(f'{k}: {v}')

    experiment.execute()
