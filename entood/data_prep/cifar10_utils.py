
import tensorflow as tf
import tensorflow_datasets as tfds


def preproc(image, label):
    x = tf.cast(image, tf.float32) / 255.
    y = tf.one_hot(label, 10)
    return x, y


def preproc_images(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return image


def load_images(shuffle=False, take=None):
    datasets, ds_info = tfds.load('cifar10', data_dir='data/cifar10',
                                  shuffle_files=shuffle, as_supervised=True, with_info=True)
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc_images)
    test_dataset = datasets[tfds.Split.TEST].map(preproc_images)
    if take:
        train_dataset = train_dataset.take(take)
        test_dataset = test_dataset.take(take)

    return (train_dataset, test_dataset), ds_info


def load_data():
    datasets = tfds.load('cifar10', data_dir='data/cifar10',
                         shuffle_files=True, as_supervised=True)

    train_dataset = datasets[tfds.Split.TRAIN].map(preproc)
    test_dataset = datasets[tfds.Split.TEST].map(preproc)

    return train_dataset, test_dataset
