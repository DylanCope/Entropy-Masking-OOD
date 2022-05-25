"""
Script to analyse pretrained describers
"""

import argparse
from functools import partial
import gc
import os
import sys
from typing import List, Dict, Callable

import tensorflow as tf

import pandas as pd

from analogyproj.analysis.describer_masking import MaskAccuracyComputer
from analogyproj.analysis.describers_utils import DescribersLoader
from analogyproj.data_prep import image_loaders
from analogyproj.data_prep.k_constrast_ds import k_contrast_desc_ds, k_to_batch_size

tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_contrastive_datasets(
        ks: List[int],
        image_datasets: Dict[str, Callable]
) -> Dict[str, Dict[int, dict]]:
    """
    Args:
        ks: List of k values to use for each contrastive ds
        image_datasets: Image datasets to use as for contrastive datasets, dataset name -> dataset loader function
        lazy: load dataset now or later? If true, also assumes image_datasets is lazy

    Returns:
        A dictionary from dataset names to dicts of ks to callables to get dataset item
    """
    contrastive_datasets = dict()
    
    def load(ds_name, k):
        get_ds = image_datasets[ds_name]
        (train_imgs, test_imgs), ds_info = get_ds()
        
        batch_size = k_to_batch_size[k]
        train_imgs_ds = train_imgs.batch(batch_size, drop_remainder=True)
        test_imgs_ds = test_imgs.batch(batch_size, drop_remainder=True)

        print(f'Creating {k}-contrastive dataset for {ds_name}')
        train_k_contrast_ds = k_contrast_desc_ds(train_imgs_ds, k=k)
        test_k_contrast_ds = k_contrast_desc_ds(test_imgs_ds, k=k)
        
        return {
            'train_ds': train_k_contrast_ds, 
            'val_ds': test_k_contrast_ds,
            'batch_size': batch_size,
            'batches_per_epoch': len(train_imgs_ds),
            'val_steps': len(test_imgs_ds),
            'base_dataset': ds_name,
        }
    
    return {
        ds_name: {
            k: partial(load, ds_name, k)
            for k in ks
        }
        for ds_name in image_datasets
    }


def _try_load_existing_items(df_path: str, overwrite_files: bool) -> List[dict]:
    try:
        df = pd.read_csv(df_path)
        df = df.drop('Unnamed: 0', axis=1, errors='ignore')

        print(f'DataFrame already exists at {df_path}')
        if not overwrite_files:
            print('Refusing to overwrite!')
            sys.exit(0)

        print('Loading items from existing df')
        return df.to_dict('records')

    except FileNotFoundError:
        return []


def compute_masked_accuracies(
        args,
        describers,
        image_datasets,
        contrastive_datasets
):
    masked_acc_df_path = args.masked_acc_df_path
    
    # These datasets do not have notable shift from CIFAR-10
    skip_datasets = ['Stanford Online Products', 'CIFAR-100']
    if not args.cf10_accs:
        skip_datasets.append('CIFAR-10')

    masked_items = _try_load_existing_items(masked_acc_df_path, args.overwrite_files)

    if args.mask_prop_sweep:
        print('Sweeping over mask proportions:', ', '.join(map(str, args.mask_prop_sweep)))
    
    eval_datasets = sorted(set(contrastive_datasets) - set(skip_datasets))
    
    print('Evaluating masked accuracies on datasets:', ', '.join(eval_datasets))

    if not eval_datasets:
        raise ValueError('No evaluation datasets provided')

    print('Will write results to:', masked_acc_df_path)
    masked_acc_df = pd.DataFrame(masked_items)
    
    if args.cf10_entropy_mask:
        cf10_images = image_datasets['CIFAR-10']()
        print('Using CIFAR-10 images to compute entropy values')
    
    for ds_name in eval_datasets:

        print(f'Evaluating {ds_name} masked accuracies...')
        datasets_loaders = contrastive_datasets[ds_name]
        datasets = {k: loader() for k, loader in datasets_loaders.items()}
        

        if args.mask_prop_sweep:
            name = f'masked_accuracies_sweep_{ds_simple_name(ds_name)}'
        else:
            name = f'masked_accuracies_{ds_simple_name(ds_name)}'
            
        if args.cf10_entropy_mask:
            (_, base_dataset), _ = cf10_images
            name += '_cf10_ent' 
        else:
            (_, base_dataset), _ = image_datasets[ds_name]()

        if masked_items:
            ds_items = [
                item for item in masked_items
                if item['Dataset'] == ds_name
            ]
        else:
            ds_items = None
        
        masked_acc_computer = MaskAccuracyComputer(describers, base_dataset, datasets,
                                                   mask_prop=args.mask_prop, 
                                                   mask_prop_sweep=args.mask_prop_sweep,
                                                   items=ds_items,
                                                   max_eval_steps=args.max_eval_steps,
                                                   name=name)
        try:
            masked_acc_computer.run()
        
        finally:
            masked_acc_computer.save()
            masked_items.extend(masked_acc_computer.items)
            masked_acc_df = pd.DataFrame(masked_items)
            masked_acc_df.to_csv(masked_acc_df_path)
            
            del datasets, masked_acc_computer, base_dataset
            gc.collect()

    print(f'Finished computing masked accuracies.')
    
    return masked_acc_df


def ds_simple_name(ds_name):
    name = ds_name.lower()
    name = name.replace(' ', '_')
    name = name.strip()
    name = name.replace('dataset', '')
    name = name.replace('-', '')
    return name


def main(args):
    ds_names = '\n - '.join((args.datasets or image_loaders.keys()))
    
    eval_datasets = [
        ds_simple_name(ds) 
        for ds in (args.datasets or image_loaders.keys())
    ]
    
    print('Loading datasets:', ds_names)
    
    image_datasets = {
        ds_name: load_images
        for ds_name, load_images in image_loaders.items()
        if ds_simple_name(ds_name) in eval_datasets
    }
    
    loader = DescribersLoader()
    if args.num_describers:
        import itertools
        describers = list(itertools.islice(loader.load_describers(as_list=False, shuffle=True), 
                                           int(args.num_describers)))
    else:
        describers = loader.load_describers()
    
    ks = loader.datasets.keys()
    contrastive_datasets = get_contrastive_datasets(ks, image_datasets)
    
    if args.task == 'accs':
        # compute_accuracies(args, describers, image_datasets, contrastive_datasets)
        pass
    elif args.task == 'masked_accs':
        compute_masked_accuracies(args, describers, image_datasets, contrastive_datasets)


def parse_args():
    parser = argparse.ArgumentParser()

    # default hyperparams chosen from sweep on baseline_ft_ae
    parser.add_argument('--task', type=str, default='masked_accs',
                        help='Which analysis task to perform (masked_accs, accs)')
    
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to evaluate')
    parser.add_argument('--mask_prop_sweep', type=float, nargs='+', default=None,
                        help='Mask prop values to sweep')
    parser.add_argument('--mask_prop', type=float, default=0.3,
                        help='Mask prop values to sweep')
    parser.add_argument('--max_eval_steps', type=int, default=500,
                        help='Number of evaluation batches to take')
    parser.add_argument('--cf10_entropy_mask', default=False, action='store_true',
                        help='Whether or not to use CIFAR-10 dataset for entropy computation')
    parser.add_argument('--cf10_accs', default=False, action='store_true',
                        help='Whether or not to compute accuracies for CIFAR-10 dataset')
    parser.add_argument('--num_describers', default=None, help='How many describers to load')
    
    parser.add_argument('--masked_acc_df_path', type=str,
                        default='experiments/contrastive/describer_masked_accuracies.csv',
                        help='Where to write masked accuracy DataFrame to.')
    parser.add_argument('--overwrite_files', default=False, action='store_true',
                        help='Whether or not to overwrite an existing file.')
    
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
