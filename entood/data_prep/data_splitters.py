import abc
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class DataSplitStrategy(abc.ABC):
    
    def __init__(self, 
                 train_ds: tf.data.Dataset,
                 test_ds: tf.data.Dataset, 
                 batch_size: int,
                 seed: int = None):
        '''
        Args:
            train_ds (tf.data.Dataset): Test dataset Unbatched an no prefetch
            test_ds (tf.data.Dataset): Train dataset. Unbatched and no prefetch
            batch_size (int): Batch size
        '''
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        
        self.seed = seed
        if seed:
            np.random.seed(seed)
            
    def get_config(self):
        # must be defined in subclasses
        in_distribution_classes = self.in_distribution_classes.numpy().tolist()
        out_distribution_classes = self.out_distribution_classes.numpy().tolist()
        return {
            'in_distribution_classes': in_distribution_classes,
            'out_distribution_classes': out_distribution_classes,
            'batch_size': self.batch_size,
            'seed': self.seed,
        }
    
    def batch_and_prefetch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def get_in_distribution_train_ds(self) -> tf.data.Dataset:
        return self.get_in_distribution(self.train_ds)
    
    def get_in_distribution_test_ds(self) -> tf.data.Dataset:
        return self.get_in_distribution(self.test_ds)
    
    def get_out_distribution_train_ds(self) -> tf.data.Dataset:
        return self.get_out_distribution(self.train_ds)
    
    def get_out_distribution_test_ds(self) -> tf.data.Dataset:
        return self.get_out_distribution(self.test_ds)
        
    def filter_by_classes(self, 
                          dataset: tf.data.Dataset, 
                          classes: tf.Tensor) -> tf.data.Dataset:
        
        @tf.function
        def filter_fn(features):
            label = tf.argmax(features['label'])
            return tf.reduce_any(tf.equal(label, classes))
        
        dataset = dataset.filter(filter_fn)
        return self.batch_and_prefetch(dataset)
    
    def get_in_distribution(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return self.filter_by_classes(dataset, self.in_distribution_classes)
    
    def get_out_distribution(self, dataset) -> tf.data.Dataset:
        return self.filter_by_classes(dataset, self.out_distribution_classes)
    

class HoldOutMetaClassesSplitStrategy(DataSplitStrategy):
    '''
    Chooses a split of metaclasses to be in or out of distribution.
    For each split it gives all of the classes within each given meta class. 
    '''
    
    def __init__(self, 
                 in_dist_prop: float, 
                 n_coarse_labels: int, 
                 *args, **kwargs):
        super(HoldOutMetaClassesSplitStrategy, self).__init__(*args, **kwargs)
        
        # needs fixing (in distribution classes should not be meta classes)
        meta_classes = list(range(n_coarse_labels))
        np.random.shuffle(meta_classes)
        i = int(in_dist_prop * n_coarse_labels)
        self.in_distribution_classes = tf.convert_to_tensor(meta_classes[:i], tf.int64)
        self.out_distribution_classes = tf.convert_to_tensor(meta_classes[i:], tf.int64)
        

class HoldOutFineClassesSplitStrategy(DataSplitStrategy):
    '''
    For each metaclass choose fine subclasses to be in or out of distribution.
    '''
    
    def __init__(self, 
                 in_dist_prop: float, 
                 n_coarse_labels: int, 
                 type_mapping: Dict[int, List[int]], 
                 *args, **kwargs):
        
        super(HoldOutFineClassesSplitStrategy, self).__init__(*args, **kwargs)
        
        meta_classes = list(range(n_coarse_labels))
        
        self.in_distribution_classes = []
        self.out_distribution_classes = []
        
        for coarse_label in meta_classes:
            
            fine_classes = type_mapping[coarse_label]
            i = int(in_dist_prop * len(fine_classes))
            
            np.random.shuffle(fine_classes)
            
            self.in_distribution_classes.extend(fine_classes[:i])
            self.out_distribution_classes.extend(fine_classes[i:])
            
        self.in_distribution_classes = tf.convert_to_tensor(self.in_distribution_classes, tf.int64)
        self.out_distribution_classes = tf.convert_to_tensor(self.out_distribution_classes, tf.int64)
