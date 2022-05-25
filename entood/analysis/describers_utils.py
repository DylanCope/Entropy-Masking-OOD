from pathlib import Path
import yaml
import json
import random
import time

import pandas as pd
import numpy as np
import tensorflow as tf

from analogyproj.data_prep.k_constrast_ds import k_contrast_desc_ds, k_to_batch_size
from analogyproj.data_prep import image_loaders
from analogyproj.models.contrastive_desc_learning import ContrastiveDescriptionLearner
from analogyproj.utils import progress_bar_string


def binary_entropy(x, eps=1e-12):
    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[-1]
    p = tf.reduce_sum(x, axis=-1) / tf.cast(n, tf.float32)
    
    pos_term = - p * tf.experimental.numpy.log2(p + eps) 
    neg_term = - (1 - p) * tf.experimental.numpy.log2(1 - p + eps)
    
    return pos_term + neg_term


def compute_entropy_df(describers, dataset, silent=False, **pred_kwargs):
    entropy_items = []
    total = len(describers)
    for i, describer in enumerate(describers): 
        k = describer.training_k
        descs = describer.answer_encoder.predict(dataset, **pred_kwargs)
        
        entropies = binary_entropy(tf.transpose(descs)).numpy()
        n = int(describer.encoding_size)
        index = np.arange(n)
        original_pos = 100 * index / (n - 1)
        sorted_pos = 100 * entropies.argsort().argsort() / (n - 1)

        entropy_items.extend([
            {'Describer ID': hash(describer), 'Number of Distractors': k, 'Description Length': n,
             'Entropy': e, 'Index': i, 'Position': x, 'Percentile': x_sorted}
            for i, e, x, x_sorted in zip(index, entropies, original_pos, sorted_pos)
        ])
        if not silent:
            print(progress_bar_string(i+1, total), end='\r')
    
    if not silent:
        print(progress_bar_string(total, total), end='\r')
    
    return pd.DataFrame(entropy_items)


def load_config(wandb_run_path):
    try:
        with open(wandb_run_path / 'files/config.yaml') as f:
            return {
                k: v.get('value')
                for k, v in yaml.full_load(f).items()
                if isinstance(v, dict) and not k.startswith('_')
            }
    except FileNotFoundError:
        return None

    
def load_summary(wandb_run_path):
    try:
        with open(wandb_run_path / 'files/wandb-summary.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

    
def untrained_loss(k, n=100000):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    y_true = np.random.randint(0, k, size=n)
    y_pred = np.random.uniform(0, 1, (n, k))
    return scce(y_true, y_pred).numpy()
    

class DescribersLoader:
    
    def __init__(self, 
                 images_ds_name='CIFAR-10', 
                 loss_cuttoff_p=0.8,
                 min_train_epochs=10, 
                 max_train_epochs=10):
    
        self.datasets = dict()
        self.base_images_ds_name = images_ds_name
        self.base_images_ds, _ = image_loaders[images_ds_name]() # (train_images, test_images)
        
        self.min_train_epochs = min_train_epochs
        self.max_train_epochs = max_train_epochs
        
        self.loss_cutoff_p = loss_cuttoff_p
        
        self.total_to_load = 0
        self.num_processed = 0
        self.time_start = None
    
    def is_untrained(self, describer):
        threshold_loss = untrained_loss(describer.training_k) * self.loss_cutoff_p
        return describer.training_summary['loss'] > threshold_loss
        
    
    def load_describers(self, filter_untrained=True, as_list=True, verbose=False, shuffle=False):
        
        paths = [
            (run_path, config)
            for run_path in Path('wandb').glob('run-*')
            if (config := load_config(run_path)) 
            and config.get('name', '').startswith('describer')
        ]
        
        if shuffle:
            random.shuffle(paths)
        
        self.total_to_load = len(paths)
        self.num_processed = 0
        self.time_start = time.time()
        print('Loading describers...')
        print(progress_bar_string(self.num_processed, self.total_to_load), end='\r')
        
        describers = (
            describer
            for (run_path, config) in paths
            if (describer := self.load_describer(run_path, config, verbose=verbose))
        )

        if filter_untrained:
            describers = (
                describer for describer in describers
                if not self.is_untrained(describer)
            )

        if as_list:
            describers = list(describers)
            if not verbose:
                time_elapsed = time.time() - self.time_start
                print(progress_bar_string(self.num_processed, self.total_to_load, 
                                          time_elapsed=time_elapsed), 
                      end='\n')
            
        return describers
        

    def load_describer(self, run_path, config, verbose=False):

        model_config = config.get('model_config')
        if not model_config or model_config == 'N/A':
            self.num_processed += 1
            return None

        max_epochs = config.get('max_epochs', 0)
        if max_epochs < self.min_train_epochs or max_epochs > self.max_train_epochs:
            self.num_processed += 1
            return None
        
        if verbose:
            print(config)

        summary = load_summary(run_path)
        
        if verbose:
            print(run_path, summary)

        enc_size = config['model_config']['desc_len']
        name = config['model_config']['name']

        ds_info = config.get('ds_info') or {'num_distractors': 3}
        k = ds_info.get('num_distractors', 3)

        if k not in self.datasets:
            batch_size = k_to_batch_size[k]
            train_imgs, test_imgs = self.base_images_ds
            train_imgs_ds = train_imgs.batch(batch_size, drop_remainder=True)
            test_imgs_ds = test_imgs.batch(batch_size, drop_remainder=True)
            if verbose:
                print(f'Creating {k}-contrastive dataset')
            
            train_k_contrast_ds = k_contrast_desc_ds(train_imgs_ds, k=k)

            test_k_contrast_ds = k_contrast_desc_ds(test_imgs_ds, k=k)

            self.datasets[k] = {
                'train_ds': train_k_contrast_ds, 
                'val_ds': test_k_contrast_ds,
                'batch_size': batch_size,
                'batches_per_epoch': len(train_imgs_ds),
                'val_steps': len(test_imgs_ds),
                'base_dataset': self.base_images_ds_name,
            }

        try:
            describer = ContrastiveDescriptionLearner(enc_size=enc_size)
            describer.training_k = k
            describer.training_summary = summary
            describer.training_ds = self.datasets[k]
            identifier = hash((k, enc_size, summary['_runtime'], summary['_timestamp'], summary['loss']))
            describer.backup_id = summary.get('graph', {'sha256': identifier}).get('sha256', identifier)
            describer.identifier = identifier
            build_input, _ = next(iter(describer.training_ds['train_ds']))
            describer(build_input, training=True)
            if verbose:
                print(f'Loading ContrastiveDescriptionLearner ({enc_size=}, {describer.training_k=}) from {run_path}')
                print()
            describer.load_weights(f'{run_path}/files/model-best.h5')
            describer.compile(metrics=['acc'])
            return describer
            
        except FileNotFoundError:
            print('Failed to find model-best weights')
        
        finally:
            self.num_processed += 1
            if not verbose:
                time_elapsed = time.time() - self.time_start
                print(progress_bar_string(self.num_processed, self.total_to_load, time_elapsed=time_elapsed), 
                      end='\r')

                
                