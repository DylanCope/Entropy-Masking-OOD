import tensorflow as tf


class ParallelApply(tf.keras.Model):

    def __init__(self, submodel, **kwargs):
        super(ParallelApply, self).__init__(**kwargs)
        self.submodel = submodel

    def call(self, inputs, training=False):
        return tf.map_fn(lambda z: self.submodel(z, training),
                         inputs)
