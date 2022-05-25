import tensorflow as tf
import tensorflow_datasets as tfds


def preproc(features):
    features['image'] = tf.image.resize(
                           features['image'],
                           [32,32], #height and width
                           method=tf.image.ResizeMethod.BILINEAR,
                           preserve_aspect_ratio=False,
                           antialias=False,
                            # name=None
                          )
    features['image'] = tf.cast(features['image'], tf.float32) / 255.
    features['super_class_id'] = tf.one_hot(features['super_class_id'], 12) # there are 22634 fine classes
    return features


def preproc_images(features):
    features['image'] = tf.image.resize(
                           features['image'],
                           [32,32], #height and width
                           method=tf.image.ResizeMethod.BILINEAR,
                           preserve_aspect_ratio=False,
                           antialias=False,
                            # name=None
                          )
    features['image'] = tf.cast(features['image'], tf.float32) / 255.
    return features['image']


def load_images(take=None):
    datasets, ds_info = tfds.load('StanfordOnlineProducts', data_dir='image_classification.StanfordOnlineProducts',
                                  shuffle_files=False, as_supervised=False, with_info=True)
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc_images)
    test_dataset = datasets[tfds.Split.TEST].map(preproc_images)
    if take:
        train_dataset = train_dataset.take(take)
        test_dataset = test_dataset.take(take)

    return (train_dataset, test_dataset), ds_info


def load_data():
    datasets = tfds.load('StanfordOnlineProducts', data_dir='image_classification.StanfordOnlineProducts',
                         shuffle_files=True, as_supervised=False)
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc)
    test_dataset = datasets[tfds.Split.TEST].map(preproc)

    return train_dataset, test_dataset
