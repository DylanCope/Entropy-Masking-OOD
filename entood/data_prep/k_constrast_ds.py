import tensorflow as tf


k_to_batch_size = {
    20: 16,
    10: 32,
    5: 64,
    3: 64
}


def k_contrast_desc_ds(ds: tf.data.Dataset,
                       k: int = 5,
                       rng_seed: int = 0,
                       buffer_size: int = 100):
    if k <= 0 or type(k) != int:
        raise ValueError(f'Invalid argument {k=}, k must be an integer greater than zero')

    shuffled_datasets = tuple([
        ds.shuffle(buffer_size,
                   seed=rng_seed + i,
                   reshuffle_each_iteration=True).repeat()
        for i in range(k)
    ])
    zipped_ds = tf.data.Dataset.zip(shuffled_datasets)
    rng = tf.random.Generator.from_seed(rng_seed, alg='philox')

    k_indices = tf.convert_to_tensor(range(k), dtype=tf.int64)

    def make_contrast_desc_item(*xs):
        xs = tf.stack(xs)
        batch_size = tf.shape(xs)[1]
        idx = rng.uniform_full_int([batch_size, 1], dtype=tf.int64) % k
        problem = tf.einsum('ib...->bi...', xs)
        answer = problem[idx == k_indices]
        feats = {
            'problem': problem,
            'answer': answer
        }
        return feats, idx

    kcd_ds = zipped_ds.map(make_contrast_desc_item,
                           num_parallel_calls=tf.data.AUTOTUNE)

    return kcd_ds
