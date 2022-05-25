import json
from pathlib import Path
import time
from collections import defaultdict
from typing import List, Dict, Callable, Union, Optional, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import wandb
from .config import RUNS_DIR

sns.set()


def training_step(model: tf.keras.Model,
                  inputs: tf.Tensor,
                  y_true: tf.Tensor,
                  optimizer: Optional[tf.optimizers.Optimizer] = None,
                  loss_fn: Optional[Callable] = None,
                  trainable_vars: Optional[List[tf.Variable]] = None,
                  metrics_fn: Optional[List[Callable]] = None) -> Dict[str, tf.Tensor]:
    """
    Perform a single step of optimisation on a given model

    Args:
        model: Model to be optimised
        inputs: Inputs to be given
        y_true: Targets to use to judge the output of the model
        optimizer: Optimizer to perform the update
                   If not provided it is assumed the model was compiled with an optimizer
        loss_fn: Function to judge the output: loss_fn(y_true, y_pred)
                 If not provided it is assumed the model was compiled with a loss function
        trainable_vars:
        metrics_fn:

    Returns: The metrics gathered from this training step.

    """
    optimizer = optimizer or model.optimizer
    loss_fn = loss_fn or model.compiled_loss
    trainable_vars = trainable_vars or model.trainable_variables
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=True)
        loss = loss_fn(y_true, y_pred)
        gradients = tape.gradient(loss, trainable_vars)

    optimizer.apply_gradients(zip(gradients, trainable_vars))

    if model.compiled_metrics:
        model.compiled_metrics.update_state(y_true, y_pred)
        metrics = {
            metric.name: metric.result()
            for metric in model.compiled_metrics.metrics
        }
    elif metrics_fn:
        metrics = metrics_fn(y_pred, y_true)
    else:
        metrics = dict()

    return {'loss': loss, **metrics}


def time_id():
    return int(time.time() * 1e7)


class Run:

    def __init__(self,
                 model,
                 train_ds,
                 ds_info=None,
                 loss_fn=None,
                 val_ds=None,
                 optimiser=None,
                 lr=0.001,
                 num_epochs=10,
                 batches_per_epoch=-1,
                 val_steps=-1,
                 custom_training_step=None,
                 do_analysis=True,
                 custom_analyse=None,
                 model_type='classifier',
                 val_freq=1,
                 metric_fns: List[Union[tf.keras.metrics.Metric, str, Callable]] = None,
                 callbacks: List[tf.keras.callbacks.Callback] = None,
                 run_dir=None,
                 keras_fit=False,
                 end_wandb=True,
                 verbose=1,
                 wandb_entity='dydi-analogies',
                 wandb_project='project',
                 name='run'):

        self.run_id = time_id()
        self.name = name
        self.model = model
        self.ds_info = ds_info
        self.epoch = 0
        self.train_ds = train_ds.repeat()
        self.val_freq = val_freq
        self.val_ds = val_ds.repeat()
        self.training_step = custom_training_step or tf.function(training_step)
        self.optimiser = optimiser or tf.keras.optimizers.Adam(learning_rate=lr)
        self.metric_fns = metric_fns or model.compiled_metrics or []
        self.run_dir = run_dir or f'{RUNS_DIR}/{model.name}/{self.name}-{self.run_id}'
        self.model_weights_dir = f'{self.run_dir}/model_weights'
        self.model_type = model_type
        self.do_analysis = do_analysis
        self.custom_analyse = custom_analyse or None
        self.loss_fn = loss_fn or model.compiled_loss
        self.num_epochs = num_epochs
        self.history = None
        self.verbose = verbose
        
        Path(self.run_dir).mkdir(parents=True)
        Path(self.model_weights_dir).mkdir(parents=True)

        self.end_wandb = end_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        if batches_per_epoch > 0:
            self.batches_per_epoch = batches_per_epoch
        else:
            self.batches_per_epoch = len(train_ds)

        if val_steps > 0:
            self.val_steps = val_steps
        elif val_ds:
            self.val_steps = len(val_ds)
        else:
            self.val_steps = None

        self.keras_fit = keras_fit
        self.model.compile(loss=loss_fn,
                           optimizer=self.optimiser,
                           metrics=metric_fns)

        self.build()

        self.user_given_callbacks = callbacks or []
        self.callbacks = None  # To be defined in pre execution setup

    def build(self):
        print(f'Building model: {self.model}')
        x, y_true = next(iter(self.train_ds))
        y_pred = self.model(x, training=True)
        self.loss_fn(y_true, y_pred)

    def get_config(self):
        try:
            model_config = self.model.get_config()
        except NotImplementedError:
            model_config = None

        trainable_vars = self.model.trainable_variables or []
        trainable_var_names = [v.name for v in trainable_vars]
        metrics = [str(f) for f in self.metric_fns]

        return {
            'name': self.name,
            'run_dir': self.run_dir,
            'model_type': self.model_type,
            'optimiser_config': self._clean_for_json(self.optimiser.get_config()),
            'model_config': self._clean_for_json(model_config),
            'max_epochs': self.num_epochs,
            'loss_fn': str(self.loss_fn),
            'metrics': metrics,
            'ds_info': self.ds_info,
            'train_steps_per_epoch': self.batches_per_epoch,
            # 'callback_params': self.callbacks.params,
        }

    def update_model(self, inputs, targets):

        logs = self.training_step(self.model,
                                  inputs,
                                  targets,
                                  self.optimiser,
                                  self.loss_fn,
                                  self.model.trainable_variables)

        if not isinstance(logs, dict):
            logs = {'loss': logs}  # assume it is a loss value
        return logs

    def train(self):
        """
        Executes a custom training loop.
        """
        self.callbacks.on_train_begin()
        while self.epoch < self.num_epochs:
            self.model.compiled_metrics.reset_state()
            self.callbacks.on_epoch_begin(self.epoch)

            self.epoch = self.epoch + 1
            train_steps = self.train_ds.take(self.batches_per_epoch)
            epoch_sum_loss = tf.zeros(1)
            for step, (inputs, targets) in enumerate(train_steps):
                self.callbacks.on_train_batch_begin(step)

                logs = self.update_model(inputs, targets)
                epoch_sum_loss += logs['loss']
                logs['loss'] = epoch_sum_loss / (step + 1)

                self.callbacks.on_train_batch_end(step, logs)
            epoch_logs = logs

            if self.val_steps and self.epoch % self.val_freq == 0:
                val_logs = self.model.evaluate(
                    self.val_ds,
                    steps=self.val_steps,
                    callbacks=self.callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True)
                val_logs = {'val_' + name: val for name, val in val_logs.items()}
                epoch_logs.update(val_logs)

            self.callbacks.on_epoch_end(self.epoch, epoch_logs)

        self.callbacks.on_train_end()
        self.history = self.model.history.history

    def plot_loss_curve(self, savefig=True, showfig=False, figsize=(12, 7)):
        fig, ax = plt.subplots(figsize=figsize)
        df = pd.DataFrame([
            {'epoch': epoch, 'loss': loss, 'val_loss': val_loss}
            for epoch, (loss, val_loss) in enumerate(zip(self.history['loss'],
                                                         self.history['val_loss']))
        ])
        sns.lineplot(data=df, x='epoch', y='loss', ax=ax, label='Train Loss')
        sns.lineplot(data=df, x='epoch', y='val_loss', ax=ax, label='Validation Loss')

        wandb.log({'loss_fig': plt})

        if savefig:
            plt.savefig(f'{self.run_dir}/loss_fig.png')
        if showfig:
            plt.show()

    def analyse(self):
        self.plot_loss_curve(savefig=True, showfig=False)

        if self.custom_analyse:
            self.custom_analyse(self)

        if self.model_type == 'classifier':
            self._classification_analysis()
        elif self.model_type == 'autoencoder':
            self._autoencoder_analysis()

    def _classification_analysis(self):
        pass

    def _autoencoder_analysis(self):
        pass

    def _pre_execute_setup(self):
        self.wandb_run = wandb.init(project=self.wandb_project,
                                    entity=self.wandb_entity,
                                    config=self.get_config())

        callbacks = self.user_given_callbacks
        if not any(isinstance(c, wandb.keras.WandbCallback) for c in callbacks):
            callbacks.append(wandb.keras.WandbCallback(save_weights_only=True))
        if not any(isinstance(c, tf.keras.callbacks.ModelCheckpoint) for c in callbacks):
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(self.run_dir,
                                                                save_weights_only=True))
        self.callbacks = tf.keras.callbacks.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=self.verbose != 0,
            model=self.model,
            verbose=self.verbose,
            epochs=self.num_epochs,
            steps=self.batches_per_epoch
        )

    def execute(self):
        self._pre_execute_setup()

        if self.keras_fit:
            self.history = self.model.fit(
                self.train_ds,
                epochs=self.num_epochs,
                steps_per_epoch=self.batches_per_epoch,
                validation_data=self.val_ds,
                validation_steps=self.val_steps,
                callbacks=self.callbacks
            ).history
            print('Run Complete.')
            if self.end_wandb:
                self.wandb_run.finish()
                print('Run registered with W&B')
        else:
            try:
                self.train()
                if self.do_analysis:
                    self.analyse()
                self.wandb_run.finish()
                print('Run Complete.')
            except KeyboardInterrupt:
                print('Run Interrupted!')

        self.save()
        return self.history

    def save(self):

        config = {
            'ds_info': self.ds_info,
            'run_config': self.get_config(),
            'history': self._clean_for_json(self.history)
        }

        with open(f'{self.run_dir}/config.json', mode='w') as f:
            print(f'Saving run config to {f}')
            json.dump(config, f, indent=4, separators=(", ", ": "), sort_keys=True)

        print(f'Saving model weights to {self.run_dir}/model')
        self.model.save_weights(f'{self.run_dir}/model')

    @staticmethod
    def _clean_for_json(item):
        if item is None:
            return 'N/A'
        elif type(item) in [str, int, float, bool]:
            return item
        elif isinstance(item, tf.Tensor):
            # assuming 1D tensor
            return float(item.numpy())
        elif isinstance(item, list):
            return [Run._clean_for_json(x) for x in item]
        elif type(item) in [np.float32, np.float32]:
            return float(item)
        elif type(item) in [np.int32, np.int64]:
            return int(item)
        elif isinstance(item, tuple):
            return tuple([Run._clean_for_json(x) for x in item])
        elif type(item) in [dict, defaultdict]:
            return {Run._clean_for_json(k): Run._clean_for_json(v)
                    for k, v in item.items()}

        try:
            return str(item)
        except Exception:
            raise ValueError(f'Unexpected item type in history: {item=}')


class RunSequence:
    NOT_STARTED = 'Not Started'
    IN_PROGRESS = 'In Progress'
    COMPLETE = 'Complete'

    def __init__(self,
                 runs: Union[Generator[Run, None, None], List[Run]],
                 name='runs'):
        self.name = name
        
        self.lazy = isinstance(runs, Generator)
        
        self.runs = (
            {'index': i, 'run': run, 'status': RunSequence.NOT_STARTED}
            for i, run in enumerate(runs)
        )
        if not self.lazy:
            self.runs = list(self.runs)
            self.num_runs = len(runs)
        else:
            self.num_runs = None

    def get_config(self):
        if not self.lazy:
            return {
                run_item['index']: {
                    'run_config': run_item['run'].get_config(),
                    'status': run_item['status'],
                }
                for run_item in self.runs
            }
        else:
            return {}

    def execute(self):
        for run_item in self.runs:
            run = run_item['run']
            i = run_item['index']
            print(f'Executing run {i}: {run.name}')
            
            if not self.lazy:
                self.runs[i]['status'] = RunSequence.IN_PROGRESS
            
            run.execute()
            
            if not self.lazy:
                self.runs[i]['status'] = RunSequence.COMPLETE
            
            # if self.lazy:
            #     try:
            #         cuda.select_device(0)
            #         cuda.close()
            #     except:
            #         pass
