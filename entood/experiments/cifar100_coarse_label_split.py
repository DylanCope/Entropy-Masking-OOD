from ..analysis.classification import acc_metric
from ..data_prep.cifar100_utils import (
    get_cifar100_type_mapping, 
    load_data as load_cifar100_data,
    N_COARSE_LABELS as CIFAR100_N_COARSE_LABELS,
)
from ..data_prep.data_splitters import HoldOutFineClassesSplitStrategy
from ..models.baseline_classifiers import (
    ClassifierOnAE, ClassifierFineTuneAE, ClassifierNoAE
)
from ..models.basic_autoencoder import BasicAutoencoder
from ..models.platostore import (
    PlatostoreAE,
    ClassifierOnPAE,
)
from ..run import Run

from pathlib import Path
import time

import tensorflow as tf
import wandb


def generate_data(config):
    train_dataset, test_dataset = load_cifar100_data()
    
    ds_splitter = HoldOutFineClassesSplitStrategy(config['in_dist_prop'],
                                                  CIFAR100_N_COARSE_LABELS,
                                                  get_cifar100_type_mapping(),
                                                  train_dataset,
                                                  test_dataset,
                                                  config['batch_size'])
    
    train_dataset_ind = ds_splitter.get_in_distribution_train_ds() 
    test_dataset_ind = ds_splitter.get_in_distribution_test_ds()
    train_dataset_ood = ds_splitter.get_out_distribution_train_ds() 
    test_dataset_ood = ds_splitter.get_out_distribution_test_ds()
    
    train_dataset_ind = train_dataset_ind.map(lambda feats: (feats['image'], feats['coarse_label']))
    test_dataset_ind = test_dataset_ind.map(lambda feats: (feats['image'], feats['coarse_label']))
    train_dataset_ood = train_dataset_ood.map(lambda feats: (feats['image'], feats['coarse_label']))
    test_dataset_ood = test_dataset_ood.map(lambda feats: (feats['image'], feats['coarse_label']))

    datasets = (train_dataset_ind, test_dataset_ind, train_dataset_ood, test_dataset_ood)
    
    ind_size = ds_splitter.in_distribution_classes.numpy().size
    ood_size = ds_splitter.out_distribution_classes.numpy().size
    batches_in_train = len(train_dataset) / config['batch_size']
    batches_in_test = len(test_dataset) / config['batch_size']
    
    n_train_ind_batches = int(batches_in_train * ind_size / (ind_size + ood_size))
    n_test_ind_batches = int(batches_in_test * ind_size / (ind_size + ood_size))
    n_train_ood_batches = int(batches_in_train * ood_size / (ind_size + ood_size))
    n_test_ood_batches = int(batches_in_test * ood_size / (ind_size + ood_size))
    
    ds_info = {
        'dataset_name': 'cifar100_holdout_fine',
        'batch_size': config['batch_size'],
        'in_dist_prop': config['in_dist_prop'],
        'num_classes': CIFAR100_N_COARSE_LABELS,
        'n_train_ind_batches': n_train_ind_batches,
        'n_test_ind_batches': n_test_ind_batches,
        'n_train_ood_batches': n_train_ood_batches,
        'n_test_ood_batches': n_test_ood_batches,
        **ds_splitter.get_config()
    }
    
    return datasets, ds_info


def classification_loss_fn(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
    return cce(y_true, y_pred)


def make_experiment_baseline_no_ae(train_ds, test_ds, ds_info, config,
                                   num_classes=20,
                                   **run_kwargs):
    
    clf = ClassifierNoAE(num_classes=num_classes,
                         name='baseline_no_ae')
    
    if config['optimiser'] == 'sgd':
        optimiser = tf.optimizers.SGD(learning_rate=config['learning_rate'],
                                      momentum=config['momentum'],
                                      nesterov=config['nesterov'])
    elif config['optimiser'] == 'adam':
        optimiser = tf.optimizers.Adam(learning_rate=config['learning_rate'])
    
    dataset_name = ds_info['dataset_name']
    timestamp = int(time.time() * 1e7)
    run_dir = f'./experiments/ood_experiment/{dataset_name}/{clf.name}/{timestamp}'
    Path(run_dir).mkdir(exist_ok=True, parents=True)

    callbacks = run_kwargs.pop('callbacks', []) 
    
    if config['reduce_lr_on_plateau']:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, 
                                                              patience=25))
    
    run = Run(
        clf,
        train_ds,
        optimiser=optimiser,
        loss_fn=classification_loss_fn,
        val_ds=test_ds,
        batches_per_epoch=config['batches_per_epoch'],
        val_steps=ds_info['n_test_ind_batches'],
        metric_fns=['acc'],
        num_epochs=config['num_epochs'],
        run_dir=run_dir,
        callbacks=callbacks,
        name=clf.name,
        wandb_project='cifar100_ood',
        **run_kwargs
    )

    return run


def make_experiment_baseline_ft_ae(train_ds, test_ds, ds_info, config,
                                   num_classes=20,
                                   **run_kwargs):
    ae = BasicAutoencoder()
    ae_weights_location = './models/cifar-10/basic_ae_16402223756691232'
    ae.load_weights(ae_weights_location)
    
    # build ae
    inputs, _ = next(iter(train_ds.take(1)))
    _ = ae(inputs, training=True)
    
    clf = ClassifierFineTuneAE(num_classes=num_classes,
                               pretrained_ae=ae,
                               name='baseline_ft_ae')
    
    if config['optimiser'] == 'sgd':
        optimiser = tf.optimizers.SGD(learning_rate=config['learning_rate'],
                                      momentum=0.01)
    elif config['optimiser'] == 'adam':
        optimiser = tf.optimizers.Adam(learning_rate=config['learning_rate'])
    
    dataset_name = ds_info['dataset_name']
    timestamp = int(time.time() * 1e7)
    run_dir = f'./experiments/ood_experiment/{dataset_name}/{clf.name}/{timestamp}'
    Path(run_dir).mkdir(exist_ok=True, parents=True)

    callbacks = run_kwargs.pop('callbacks', []) 
    
    if config['reduce_lr_on_plateau']:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, 
                                                              patience=25))
    
    run = Run(
        clf,
        train_ds,
        optimiser=optimiser,
        loss_fn=classification_loss_fn,
        val_ds=test_ds,
        batches_per_epoch=config['batches_per_epoch'],
        val_steps=ds_info['n_test_ind_batches'],
        metric_fns=['acc'],
        num_epochs=config['num_epochs'],
        run_dir=run_dir,
        callbacks=callbacks,
        name=clf.name,
        wandb_project='cifar100_ood',
        **run_kwargs
    )

    return run


def make_experiment_platostore_ft_pae(train_ds, test_ds, ds_info, config,
                                      num_classes=20,
                                      **run_kwargs):

    def ae_loss_fn(x, x_reconstr):
        loss = tf.keras.losses.MeanSquaredError()
        return loss(x, x_reconstr)    
    
    def classification_loss_fn(y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
        return cce(y_true, y_pred)
    
    def _platostore_joint_training_step(
        model, 
        inputs, 
        y_true, 
        *kwargs
    ):
        optimizer = model.optimizer
        trainable_vars = model.trainable_variables
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            y_pred = outputs['classifications']
            inp_reconstr = outputs['inp_reconstr']
            loss_clf = classification_loss_fn(y_true, y_pred)
            loss_ae = ae_loss_fn(inputs, inp_reconstr)
            loss = config['clf_weight'] * loss_clf + config['ae_weight'] * loss_ae
            gradients = tape.gradient(loss, trainable_vars)

        optimizer.apply_gradients(zip(gradients, trainable_vars))

        if model.compiled_metrics:
            model.compiled_metrics.update_state(y_true, y_pred)
            metrics = {
                metric.name: metric.result() 
                for metric in model.compiled_metrics.metrics
            }

        return {
            'loss': loss, 
            'loss_clf': loss_clf, 
            'loss_ae': loss_ae,
            **metrics
        }
    
    
    pae_weights_location = './models/cifar-100/pae_16427915323258540'
    pae = PlatostoreAE()
    pae.load_weights(pae_weights_location)
    
    # build pae
    inputs, _ = next(iter(train_ds.take(1)))
    _ = pae(inputs, training=True)
    
    clf = ClassifierOnPAE(pae, name='platostore_ft_pae')
    
    if config['optimiser'] == 'sgd':
        optimiser = tf.optimizers.SGD(learning_rate=config['learning_rate'],
                                      momentum=0.01)
    elif config['optimiser'] == 'adam':
        optimiser = tf.optimizers.Adam(learning_rate=config['learning_rate'])
    
    dataset_name = ds_info['dataset_name']
    timestamp = int(time.time() * 1e7)
    run_dir = f'./experiments/ood_experiment/{dataset_name}/{clf.name}/{timestamp}'
    Path(run_dir).mkdir(exist_ok=True, parents=True)

    callbacks = run_kwargs.pop('callbacks', []) 
    
    if config['reduce_lr_on_plateau']:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, 
                                                              patience=25))
    
    run = Run(
        clf,
        train_ds,
        optimiser=optimiser,
        loss_fn=classification_loss_fn,
        custom_training_step=_platostore_joint_training_step,
        val_ds=test_ds,
        batches_per_epoch=config['batches_per_epoch'],
        val_steps=ds_info['n_test_ind_batches'],
        metric_fns=['acc'],
        num_epochs=config['num_epochs'],
        run_dir=run_dir,
        callbacks=callbacks,
        name=clf.name,
        wandb_project='cifar100_ood',
        **run_kwargs
    )

    return run
    

if __name__ == '__main__':
    pass

