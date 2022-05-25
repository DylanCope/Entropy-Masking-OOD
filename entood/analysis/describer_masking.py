import json
from abc import ABC
import gc
import time
from typing import Callable, List, Union

import pandas as pd
import numpy as np
import tensorflow as tf

from analogyproj.models.contrastive_desc_learning import ContrastiveDescriptionLearner
from analogyproj.utils import progress_bar_string
from .describers_utils import compute_entropy_df


class MaskedConstrastiveModel(tf.keras.Model):
    
    def __init__(self, 
                 base_model: ContrastiveDescriptionLearner,
                 masking_strategy: Callable):
        super().__init__(name=f'masked_{base_model.name}')
        self.base_model = base_model
        self.masking_strategy = masking_strategy
        
    def call(self, feats, training=False):
        descs = self.base_model.answer_encoder(feats['answer'], training=training)
        descs = self.masking_strategy(descs)
        problem_tokens = self.base_model.problem_encoder(feats['problem'], training=training)
        return self.base_model.solve(descs, problem_tokens, training=training)
    
    
class MaskStrategy(ABC):
    
    def __init__(self, name):
        self.name = name
        self.mask = None
        
    @property
    def mask_prop(self):
        return 1 - self.mask.numpy().mean()
        
    def make_mask(self, bits_to_mask, desc_len) -> tf.Tensor:
        mask = np.ones(desc_len)
        mask[bits_to_mask] = 0.
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    
    def __call__(self, descs):
        batch_size = tf.shape(descs)[0]
        mask = tf.repeat(self.mask[tf.newaxis, :], batch_size, axis=0)
        return descs * mask + (1 - mask) * 0.5
    
    def __repr__(self):
        return f'{self.name}_{self.mask_prop:.3f}'


class MaskBottomEntropyQuantile(MaskStrategy):
    
    def __init__(self, desc_len: int, entropy_map, mask_prop=0.25, **_):
        super().__init__('Masked Bottom Entropy')
        self.entropy_map = entropy_map
        
        sorted_items = sorted(entropy_map.items(), key=lambda x: x[1])
        bottom_quartile = sorted_items[:int(mask_prop * len(sorted_items))]
        bits_to_mask = [b for b, _ in bottom_quartile]
        self.make_mask(bits_to_mask, desc_len)


class MaskTopEntropyQuantile(MaskStrategy):
    
    def __init__(self, desc_len: int, entropy_map, mask_prop=0.25, **_):
        super().__init__('Masked Top Entropy')
        self.entropy_map = entropy_map
        
        sorted_items = sorted(entropy_map.items(), key=lambda x: x[1])
        top_quartile = sorted_items[int((1 - mask_prop) * len(sorted_items)):]
        bits_to_mask = [b for b, _ in top_quartile]
        self.make_mask(bits_to_mask, desc_len)
        

class MaskRandomPercentage(MaskStrategy):

    def __init__(self, desc_len: int, mask_prop: float = 0.25, **_):
        super().__init__('Random Mask')
        
        desc_len = int(desc_len)
        bits_to_mask = np.random.choice(np.arange(desc_len), 
                                        size=int(mask_prop * desc_len),
                                        replace=False)
        self.make_mask(bits_to_mask, desc_len)
        

class NoMasking(MaskStrategy):
    
    def __init__(self, desc_len: int, **_):
        super().__init__('No Mask')
        self.make_mask([], desc_len)
        

class MaskAccuracyComputer:
    """
        Class to compute masked accuracy values for describers on contrastive datasets
    """

    def __init__(self, 
                 describers, 
                 base_dataset, 
                 datasets, 
                 mask_prop=0.5,
                 mask_prop_sweep=None,
                 use_thresholding=False,
                 strategy_cache=None,
                 min_mask_prop=0.05,
                 verbose=False,
                 silent=False,
                 max_eval_steps=500,
                 entropy_threshold=0.5,
                 items=None,
                 name='masked_accuracies',
                 base_dir='./experiments/contrastive',
                 save_mode='json',
                 save_freq=1):
        self.name = name
        self.base_dir = base_dir
        self.json_path = f'{self.base_dir}/{self.name}.json'
        self.csv_path = f'{self.base_dir}/{self.name}.csv'
        self.save_freq = save_freq
        self.save_mode = save_mode

        self.datasets = datasets
        self.base_dataset = base_dataset
        
        self.describers = describers
        
        self.mask_prop_sweep = mask_prop_sweep
        self.mask_prop = mask_prop
        self.min_mask_prop = min_mask_prop
        self.max_eval_steps = max_eval_steps
        
        self.masking_strategies = [
            MaskBottomEntropyQuantile,
            MaskTopEntropyQuantile,
            MaskRandomPercentage,
            NoMasking
        ]
        
        self.entropy_threshold = entropy_threshold
        self.use_thresholding = use_thresholding and not self.mask_prop_sweep

        self.items = items or []
        self._load_precomputed()
        if self.items:
            self.finished = [
                (item['Describer ID'], item['Masking Proportion'])
                for item in self.items
            ]
        else:
            self.finished = []

        self.silent = silent  # complete silence
        self.verbose = verbose and not silent  # only progress bar
        
        if self.mask_prop_sweep:
            self.total_to_compute = len(self.describers) * len(self.mask_prop_sweep) * len(self.masking_strategies)
        else:
            self.total_to_compute = len(self.describers) * len(self.masking_strategies)
        
        self.num_finished = 0
        self.time_start = None
        self.last_str_len = 0
        
        self.strategy_cache = strategy_cache or dict()

        self.changes_made = False

    def _compute_acc(self, describer, val_ds, val_steps, masking_strat) -> float:
        masked_describer = MaskedConstrastiveModel(describer, masking_strat)
        masked_describer.compile(metrics=['acc'])
        if self.verbose:
            print(f'Evaluating {masking_strat.name}...')
        steps = min(val_steps, self.max_eval_steps)
        _, masked_acc = masked_describer.evaluate(val_ds, steps=steps, verbose=self.verbose)
        
        self.num_finished += 1
        desc_len = describer.encoding_size
        k = describer.training_k
        self._print_progress(info_str=f'{masking_strat}, {desc_len=}, {k=}: {masked_acc=:.3f}')
        
        return masked_acc

    def _is_finished(self, describer, mask_prop, eps=0.02):
        return any(
            describer.identifier == did and np.abs(mask_prop - p) < eps
            for did, p in self.finished
        )

    def _compute_accs(self, describer, mask_prop=None):
        """
        Computes accuracies for describer with object masking strategies.
        Adds accuracies to self.items

        Args:
            describer: Describer to test accuracy of
            mask_prop: Masking proportion to use in strategies.
                If none then uses fixed mask_prop or uses thresholding
        """
        k = describer.training_k

        val_ds = self.datasets[k]['val_ds']
        val_steps = self.datasets[k]['val_steps']

        if self._is_finished(describer, mask_prop):
            self.num_finished += len(self.masking_strategies)
            self._print_progress(info_str=f'Skipping {describer.encoding_size=}, {k=}, {mask_prop=:.3f}')
            return

        if describer in self.strategy_cache and not self.mask_prop_sweep:
            masking_strats = self.strategy_cache[describer]
            mask_prop = max([strat.mask_prop for strat in masking_strats])
        else:
            entropy_df = compute_entropy_df([describer], 
                                            self.base_dataset.batch(1024), 
                                            silent=not self.verbose)
            entropy_map = dict(zip(entropy_df['Index'], entropy_df['Entropy']))

            if self.use_thresholding:
                mask_prop = np.sum(entropy_df['Entropy'] < self.entropy_threshold) / describer.encoding_size

            masking_strats = [
                masking_strat_constr(desc_len=describer.encoding_size, 
                                     mask_prop=mask_prop,
                                     entropy_map=entropy_map)
                for masking_strat_constr in self.masking_strategies
            ]

            if not self.mask_prop_sweep:
                self.strategy_cache[describer] = masking_strats

        if mask_prop < self.min_mask_prop:
            if self.verbose:
                print(f'Skipping {describer.encoding_size=} {describer.training_k=} {mask_prop=}')
                print()
            self.num_finished += len(self.masking_strategies)
            self._print_progress(info_str=f'Skipping {describer.encoding_size=}, {k=}: {mask_prop=:.3f}'
                                          '(mask prop too low)')
            return
        
        if self.verbose:
            print(f'Computing masked accuracies for {describer.encoding_size=} {describer.training_k=} {mask_prop=}')

        results = [
            (strat, self._compute_acc(describer, val_ds, val_steps, strat))
            for strat in masking_strats
        ]

        no_mask_acc: Union[float, None] = ([
            acc for (strat, acc) in results if strat.name == 'No Mask'
        ] or [None]).pop()

        self.items.extend([{
                'Describer ID': getattr(describer, 'identifier', hash(describer)),
                'Describer Backup ID': getattr(describer, 'backup_id', hash(describer)),
                'Description Length': describer.encoding_size,
                'Training $k$': describer.training_k,
                'Testing $k$': k,
                'Dataset': self.datasets[k]['base_dataset'],
                'Masking Proportion': strat.mask_prop,
                'Accuracy': acc,
                'No Mask Accuracy': no_mask_acc,
                r'$\Delta$Accuracy': acc - no_mask_acc if no_mask_acc else None,
                'Strategy': strat.name
            } for strat, acc in results
        ])

        self.changes_made = True
        self.finished.append((describer.identifier, mask_prop))
        
    def run(self, return_df=True) -> Union[pd.DataFrame, List[dict]]:
        """
        Runs the accuracy computer

        Args:
            return_df: if True returns pd.DataFrame

        Returns:
            Accuracy data in the form of a DataFrame or list of dicts
        """
        try:
            self.time_start = time.time()
            self._print_progress()
            for describer in self.describers:
                if self.mask_prop_sweep:
                    for mask_prop in self.mask_prop_sweep:
                        self._compute_accs(describer, mask_prop=mask_prop)
                        self.save()
                else:
                    self._compute_accs(describer, mask_prop=self.mask_prop)
                    self.save()

        except KeyboardInterrupt:
            print('Interrupted!')

        self._print_progress(end='\n')
        
        if return_df:
            return pd.DataFrame(self.items)
        else:
            return self.items

    def save(self):
        """
            Saves the computed accuracies to a csv
        """
        if self.changes_made:
            if self.save_mode == 'json':
                with open(self.json_path, 'w') as f:
                    json.dump(self.items, f)
                gc.collect()

    def _print_progress(self, info_str='', end='\r'):
        if not self.verbose and not self.silent:
            time_elapsed = time.time() - self.time_start
            if time_elapsed < 0.1:
                time_elapsed = None
            prog_bar = progress_bar_string(self.num_finished, self.total_to_compute,
                                           time_elapsed=time_elapsed)
            if info_str:
                prog_str = f'{prog_bar} | {info_str}'
            else:
                prog_str = prog_bar
            
            if self.last_str_len > len(prog_str):
                spaces = ' ' * (self.last_str_len - len(prog_str) + 1)
            else:
                spaces = ''
            print(prog_str + spaces, end=end)
            self.last_str_len = len(prog_str)

    def _load_precomputed(self):
        loaded_items = []

        try:
            with open(self.json_path, 'r') as f:
                loaded_items.extend(json.load(f))
            print('Retreving precomputed JSON values...')
        except FileNotFoundError:
            pass

        # try:
        #     df = pd.read_csv(self.csv_path)
        #     df.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)
        #     print('Retreving precomputed CSV values...')
        #     loaded_items.extend(df.to_dict('records'))
        # except FileNotFoundError:
        #     pass

        self.items.extend(loaded_items)
        # remove duplicates
        self.items = [
            dict(item) for item in set([tuple(record.items()) for record in self.items])
        ]
