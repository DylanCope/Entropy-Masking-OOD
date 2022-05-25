import tensorflow as tf
import tensorflow_datasets as tfds

def preproc(image, label):
    image = tf.image.resize(
                           image,
                           [32,32], #height and width
                           method=tf.image.ResizeMethod.BILINEAR,
                           preserve_aspect_ratio=False,
                           antialias=False,
                            # name=None
                          )
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(label, 39)
    return image, label


def preproc_images(image, label):
    image = tf.image.resize(
                           image,
                           [32,32], #height and width
                           method=tf.image.ResizeMethod.BILINEAR,
                           preserve_aspect_ratio=False,
                           antialias=False,
                            # name=None
                          )
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.grayscale_to_rgb(image)
    return image


def load_images(take=None):
    datasets, ds_info = tfds.load('MNIST', data_dir='image_classification.MNIST',
                                  shuffle_files=False, as_supervised=True, with_info=True)
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc_images)
    test_dataset = datasets[tfds.Split.TEST].map(preproc_images)
    if take:
        train_dataset = train_dataset.take(take)
        test_dataset = test_dataset.take(take)

    return (train_dataset, test_dataset), ds_info


def load_data():
    datasets = tfds.load('MNIST', data_dir='image_classification.MNIST',
                         shuffle_files=True, as_supervised=True)
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc)
    test_dataset = datasets[tfds.Split.TEST].map(preproc)

    return train_dataset, test_dataset
