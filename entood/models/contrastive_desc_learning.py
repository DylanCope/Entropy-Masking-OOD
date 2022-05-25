import tensorflow as tf
from official.nlp.modeling.layers import Transformer

from analogyproj.models.parallel_apply import ParallelApply
from analogyproj.models.comm_channel import CommChannel


class ContrastiveDescriptionLearner(tf.keras.Model):

    def __init__(self,
                 enc_size: int = 8,
                 prob_enc_dim: int = 128,
                 n_heads: int = 3,
                 head_dim: int = 64,
                 channel_noise: float = 0.5,
                 channel_temp: float = 1,
                 name: str = None,
                 **keras_kwargs):
        """
        Args:
            enc_size: size of channel between describer and decoder (description length)
            prob_enc_dim: size of distractor image encodings
            n_heads: number of transformer heads
            head_dim: dim of transformer heads
            channel_noise: noise between describer and decoder
            channel_temp: temperature of channel between describer and decoder
            name: name of model
            **keras_kwargs: kwargs passed to super
        """
        name = name or f'describer-{enc_size}'
        super(ContrastiveDescriptionLearner, self).__init__(name=name,
                                                            **keras_kwargs)

        self.encoding_size = int(enc_size)

        self.augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1, fill_mode='nearest'),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'),
            tf.keras.layers.RandomZoom(0.1, fill_mode='nearest'),
        ], name='augmenter')

        self.answer_encoder = tf.keras.Sequential([
            self.augmenter,
            tf.keras.layers.Conv2D(64, (3, 3),
                                   activation='relu',
                                   input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.encoding_size),
            CommChannel(enc_size, temperature=channel_temp, noise=channel_noise)
        ], name='answer_encoder')

        enc_single_problem_inp = tf.keras.Sequential([
            self.augmenter,
            tf.keras.layers.Conv2D(64, (3, 3),
                                   activation='relu',
                                   input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(prob_enc_dim),
        ], name='problem_inp_encoder')
        self.problem_encoder = ParallelApply(enc_single_problem_inp,
                                             name='problem_encoder')

        self.solver = tf.keras.Sequential([
            tf.keras.layers.Dense(head_dim * n_heads, activation='relu'),
            Transformer(n_heads, head_dim, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1),
            Transformer(n_heads, head_dim, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1),
            tf.keras.layers.Dense(1),
        ], name='solver')

        self.training_k = None

    def get_config(self) -> dict:
        return {
            'name': self.name,
            'desc_len': self.encoding_size
        }
    
    def solve(self, answer_enc, problem_tokens, training=False):
        k = tf.shape(problem_tokens)[1]
        ans_tokens = tf.repeat(answer_enc[:, tf.newaxis, :], k, axis=1)

        tokens = tf.concat([problem_tokens, ans_tokens], axis=-1)

        output = self.solver(tokens, training=training)
        output = tf.squeeze(output)
        return output
        
    def call(self, feats, training=False):
        answer = feats['answer']
        problem = feats['problem']
        answer_enc = self.answer_encoder(answer, training=training)
        problem_tokens = self.problem_encoder(problem, training=training)
        return self.solve(answer_enc, problem_tokens, training=training)
