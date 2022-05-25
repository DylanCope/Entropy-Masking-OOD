from pathlib import Path
import time

from analogyproj.models.contrastive_desc_learning import ContrastiveDescriptionLearner
from analogyproj.data_prep import load_cifar10_data
from analogyproj.data_prep.k_constrast_ds import k_contrast_desc_ds
from analogyproj import Run, RunSequence

import tensorflow as tf


def generate_data(config):
    print('Loading CIFAR-10...')
    train_dataset, test_dataset = load_cifar10_data()
    k = config['num_distractors']

    train_dataset = train_dataset.batch(config['batch_size'], drop_remainder=True)
    test_dataset = test_dataset.batch(config['batch_size'], drop_remainder=True)

    print(f'Creating {k}-contrastive data sets')
    train_imgs_ds = train_dataset.map(lambda x, y: x,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_k_contrast_ds = k_contrast_desc_ds(train_imgs_ds, k=k)

    test_imgs_ds = test_dataset.map(lambda x, y: x,
                                    num_parallel_calls=tf.data.AUTOTUNE)
    test_k_contrast_ds = k_contrast_desc_ds(test_imgs_ds, k=k)

    datasets = (train_k_contrast_ds, test_k_contrast_ds)

    ds_info = {
        'dataset_name': f'cifar10_{k}_contrast',
        'batch_size': config['batch_size'],
        'batches_per_epoch': len(train_dataset),
        'val_steps': len(test_dataset),
    }

    return datasets, ds_info


def make_experiment_vary_desc_len(train_ds, test_ds, ds_info, config, **run_kwargs):
    describers = []
    enc_sizes = [8, 16, 32, 64, 256, 512]

    for enc_size in enc_sizes:
        describer = ContrastiveDescriptionLearner(enc_size=enc_size,
                                                  channel_noise=config.get('channel_noise', 0.5))
        describers.append(describer)

    dataset_name = ds_info['dataset_name']
    timestamp = int(time.time() * 1e7)
    run_dir = f'./experiments/ood_experiment/{dataset_name}/describers/{timestamp}'
    Path(run_dir).mkdir(exist_ok=True, parents=True)

    batches_per_epoch = ds_info.get('batches_per_epoch') or config.get('batches_per_epoch') or -1
    val_steps = ds_info.get('val_steps') or config.get('val_steps') or -1

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    runs = [
        Run(describer, train_ds,
            loss_fn=loss_fn,
            val_ds=test_ds,
            run_dir=run_dir,
            wandb_project='cifar10_cdl',
            name=describer.name,
            num_epochs=config['num_epochs'],
            batches_per_epoch=batches_per_epoch,
            val_steps=val_steps,
            keras_fit=True,
            do_analysis=False)
        for describer in describers
    ]

    return RunSequence(runs)



def generate_data_vary_k(config):
    print('Loading CIFAR-10...')
    train_dataset, test_dataset = load_cifar10_data()

    ds_info = {
        'dataset_name': f'cifar10',
    }

    return (train_dataset, test_dataset), ds_info


def make_experiment_vary_k(base_train_ds, base_test_ds, ds_info, config, **run_kwargs):

    batch_sizes = {
        3: 64, 5:64, 10:32, 20:16
    }
    datasets = dict()
    
    def make_run(i, desc_len, k):
        print(f'Creating run {i} for {desc_len=}, {k=}')
        k = int(k)

        batch_size = batch_sizes[k]
        
        if k in datasets:
            train_ds = datasets[k]['train_ds']
            train_steps = datasets[k]['train_steps']
            test_ds = datasets[k]['test_ds']
            test_steps = datasets[k]['test_steps']
        else:
            datasets[k] = dict()
            
            train_dataset = base_train_ds.batch(batch_size, drop_remainder=True)
            test_dataset = base_test_ds.batch(batch_size, drop_remainder=True)
            train_steps = len(train_dataset) // k
            test_steps = len(test_dataset) // k
            
            datasets[k]['test_steps'] = test_steps
            datasets[k]['train_steps'] = train_steps

            print(f'Creating {k}-contrastive data sets')
            train_imgs_ds = train_dataset.map(lambda x, y: x,
                                              num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = k_contrast_desc_ds(train_imgs_ds, k=k)

            test_imgs_ds = test_dataset.map(lambda x, y: x,
                                            num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = k_contrast_desc_ds(test_imgs_ds, k=k)
            
            datasets[k]['test_ds'] = test_ds
            datasets[k]['train_ds'] = train_ds

        describer = ContrastiveDescriptionLearner(
            enc_size=desc_len,
            channel_noise=config.get('channel_noise', 0.5),
            name=f'describer-{desc_len}-{k}-{i}'
        )
        
        return Run(describer, train_ds,
                   loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   val_ds=test_ds,
                   wandb_project='cifar10_cdl_vary_k',
                   name=describer.name,
                   num_epochs=config['num_epochs'],
                   batches_per_epoch=train_steps,
                   ds_info={'name': 'cifar10', 'num_distractors': k, 'batch_size': batch_size},
                   val_steps=test_steps,
                   keras_fit=True,
                   do_analysis=False)
    
    runs = (
        make_run(i, desc_len, k)
        for desc_len in config['desc_lens']
        for k in config['training_ks']
        for i in range(config['repeat_runs'])
    )

    return RunSequence(runs)
