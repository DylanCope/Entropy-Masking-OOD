from typing import Dict, List, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


MAPPING_STR = '''
aquatic mammals	beaver, dolphin, otter, seal, whale
fish	aquarium fish, flatfish, ray, shark, trout
flowers	orchid, poppy, rose, sunflower, tulip
food containers	bottle, bowl, can, cup, plate
fruit and vegetables	apple, mushroom, orange, pear, sweet pepper
household electrical devices	clock, keyboard, lamp, telephone, television
household furniture	bed, chair, couch, table, wardrobe
insects	bee, beetle, butterfly, caterpillar, cockroach
large carnivores	bear, leopard, lion, tiger, wolf
large man-made outdoor things	bridge, castle, house, road, skyscraper
large natural outdoor scenes	cloud, forest, mountain, plain, sea
large omnivores and herbivores	camel, cattle, chimpanzee, elephant, kangaroo
medium mammals	fox, porcupine, possum, raccoon, skunk
non-insect invertebrates	crab, lobster, snail, spider, worm
people	baby, boy, girl, man, woman
reptiles	crocodile, dinosaur, lizard, snake, turtle
small mammals	hamster, mouse, rabbit, shrew, squirrel
trees	maple tree, oak tree, palm tree, pine tree, willow tree
vehicles 1	bicycle, bus, motorcycle, pickup truck, train
vehicles 2	lawn mower, rocket, streetcar, tank, tractor
'''

N_COARSE_LABELS = 20
N_FINE_LABELS = 100


def read_cifar100_pretty_labels() -> Tuple[List[str], List[str]]:
    data_folder = './data' # os.environ['DATA_FOLDER']
    
    with open(f'{data_folder}/cifar100/cifar100/3.0.2/label.labels.txt', mode='r') as f:
        fine_labels_names = f.read().split()
    with open(f'{data_folder}/cifar100/cifar100/3.0.2/coarse_label.labels.txt', mode='r') as f:
        coarse_labels_names = f.read().split()
        
    return fine_labels_names, coarse_labels_names


def get_pretty_cifar100_type_mapping() -> Dict[str, List[str]]:
    '''
    '''
    lines_split = [
        line_str.replace('\t\t', '\t').split('\t')
        for line_str in MAPPING_STR.split('\n')
    ][1:-1]
    
    type_mapping = {
        line[0].replace(' ', '_'): [
            fine_label.replace(' ', '_')
            for fine_label in line[1].split(', ')
        ] for line in lines_split
    }
    
    return type_mapping


def get_cifar100_type_mapping() -> Dict[int, List[int]]:
    '''
    '''
    type_mapping = get_pretty_cifar100_type_mapping()
    fine_labels_names, coarse_labels_names = read_cifar100_pretty_labels()
    
    return {
        coarse_labels_names.index(coarse_label): [
            fine_labels_names.index(fine_label)
            for fine_label in fine_labels
        ] for coarse_label, fine_labels in type_mapping.items()
    }


def preproc(features):
    features['image'] = tf.cast(features['image'], tf.float32) / 255.
    features['label'] = tf.one_hot(features['label'], N_FINE_LABELS)
    features['coarse_label'] = tf.one_hot(features['coarse_label'], N_COARSE_LABELS)
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
    datasets, ds_info = tfds.load('cifar100', data_dir='data/cifar100', 
                                  with_info=True, shuffle_files=False)
    
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc_images)
    test_dataset = datasets[tfds.Split.TEST].map(preproc_images)
    if take:
        train_dataset = train_dataset.take(take)
        test_dataset = test_dataset.take(take)
    
    return (train_dataset, test_dataset), ds_info


def load_data():
    datasets, ds_info = tfds.load('cifar100', data_dir='data/cifar100', 
                                  with_info=True, shuffle_files=True)
    
    train_dataset = datasets[tfds.Split.TRAIN].map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = datasets[tfds.Split.TEST].map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset
