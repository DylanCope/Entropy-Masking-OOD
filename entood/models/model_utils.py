import tensorflow as tf
import tensorflow_probability as tfp


def soft_discretise(x, noise=0.5, training=False):
    # DIAL DRU (discretise/regularise unit)
    if training:
        if noise > 0.0:
            dist = tfp.distributions.Normal(x, noise)
            return tf.nn.sigmoid(dist.sample())
        else:
            return tf.nn.sigmoid(x)
        
    return tf.cast(tf.nn.sigmoid(x) > 0.5, tf.float32)
