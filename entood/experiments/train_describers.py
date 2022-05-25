import argparse

import seaborn as sns

from analogyproj.experiments.describer_experiments import (
    make_experiment_vary_k as make_experiment,
    generate_data_vary_k as generate_data
)

sns.set()


def parse_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to run for.')
    parser.add_argument('--reduce_lr_on_plateau', type=bool, default=True,
                        help='Whether or not to reduce learning rate on plateau')
    parser.add_argument('--in_dist_prop', type=float, default=0.8,
                        help='Proportion of data to keep for the in-distribution.')

    # params for loading pretrained models
    parser.add_argument('--pretrained_id', type=str, default='20220330-103536',
                        help='Identifier for pretrained models to load')

    # Contrastive experiment params
    parser.add_argument('--experiment', type=str, default='vary_k',
                        help='Name of experiment to run:\n'
                             '"vary_k" - test different values of k\n'
                             '"vary_desc_len" - test different values of description length')
    parser.add_argument('--num_distractors', type=int, nargs='+', default=None,
                        help='Number of distractors (k) to use in contrastive description learning')
    parser.add_argument('--desc_len', type=int, nargs='+', default=None,
                        help='Description length to use in contrastive description learning')
    parser.add_argument('--channel_noise', type=float, default=0.5,
                        help='Amount of noise to use in channel between describer and decoder.')
    parser.add_argument('--repeat_runs', type=int, default=1,
                        help='Number of times to repeat each run')

    return vars(parser.parse_args())


def main():
    config = parse_args()
    config['desc_lens'] = config['desc_len'] or [512]
    config['training_ks'] = config['num_distractors'] or [3]

    datasets, ds_info = generate_data(config)

    for k, v in ds_info.items():
        print(f'{k}: {v}')

    experiment = make_experiment(*datasets, ds_info, config)

    for k, v in experiment.get_config().items():
        print(f'{k}: {v}')

    experiment.execute()


if __name__ == '__main__':
    main()
