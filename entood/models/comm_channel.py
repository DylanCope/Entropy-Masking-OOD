import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class CommChannel(tf.keras.layers.Layer):

    def __init__(self,
                 size=32,
                 noise=0.5,
                 temperature=1,
                 threshold=0.5,
                 no_transform=False,
                 name='comm_channel',
                 **kwargs):
        """
        Communication channel class

        Args:
            - size: number of units in channel
            - noise: standard deviation of Gaussian white noise
              in the input space of the channel.
            - temperature: sigmoidal or softmax temperature
              value that controls how sharp the distribution is
              (i.e. how close the sigmoid is to a step function
              or how close the softmax is to the max function)
            - no_transform: turn off channel transformation
        """
        super(CommChannel, self).__init__(name=name, **kwargs)
        self.size = size
        self.noise = noise
        self.temperature = temperature
        self.no_transform = no_transform
        self.threshold = threshold

    def compute_logit_delta(self, normalised_delta: float) -> float:
        """
        Compute the noise value to use to get a given amount of noise in
        the normalised space, e.g. the output, at the inflection point
        of the sigmoid curve.
        """
        two_ds = 2 * normalised_delta
        return - np.log(two_ds / (1 + two_ds)) / self.temperature

    def get_initial_state(self, batch_size):
        """ Returns output of channel when no messages have been sent yet """
        return tf.zeros((batch_size, self.size))

    def call(self, x, training=False):
        """ Send a message through the channel """

        if training:
            if self.no_transform:
                return x

            if self.noise > 0:
                x = x + tf.random.normal(tf.shape(x),
                                         mean=tf.zeros_like(x),
                                         stddev=self.noise)

            return tf.nn.sigmoid(self.temperature * x)

        else:
            return tf.cast(tf.nn.sigmoid(x) > self.threshold, tf.float32)
