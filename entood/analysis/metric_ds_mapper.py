from typing import Tuple

import tensorflow as tf


class MetricDatasetMapper:
    
    def __init__(self, 
                 metric: tf.keras.metrics.MeanMetricWrapper):
        self._metric = metric
    
    def _reduce_fn(self, _, batch: Tuple[tf.Tensor, tf.Tensor]):
        # TODO: fix this horror

        inputs, targets = batch
        preds = self._model(inputs)
        
        y_true = tf.argmax(targets, axis=-1)
        y_pred = tf.argmax(preds, axis=-1)
        
        self._metric.update_state(y_true, y_pred)

        return _
    
    def __call__(self, 
                 model: tf.keras.Model, 
                 dataset: tf.data.Dataset) -> tf.Tensor:
        self._model = model
        self._metric.reset_state()
        dataset.reduce((tf.zeros(1)), self._reduce_fn)
        return self._metric.result()
