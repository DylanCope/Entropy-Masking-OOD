import random

import tensorflow as tf
import tensorflow_addons as tfa


def data_aug(features, 
             percentage_augmented=0.1, 
             max_augmentations=3):
    
    img = features['image']
    
    possible_augmentations = []
    def augmenter(f):
        possible_augmentations.append(f)
        return f
    
    @augmenter
    def flip(img):
        flipped = tf.image.random_flip_up_down(img) #Horizontal flipping
        return flipped

    @augmenter
    def adjust_brightnes(img):
        delta = random.uniform(0, 0.4)
        bright = tf.image.adjust_brightness(img, delta)
        return bright
    
    @augmenter
    def crop(img):
        retain_frac = random.uniform(0.7, 1.0)
        cropped = tf.image.central_crop(img, central_fraction=retain_frac)
        return cropped

    @augmenter
    def rotate(img):
        radian = random.uniform(-0.3, 0.3)
        rotated = tfa.image.transform_ops.rotate(img, radian)
        return rotated
    
    
    for _ in range(random.randint(0, max_augmentations)):
        if random.random() < percentage_augmented:
            fn = random.choice(possible_augmentations)
            img = fn(img)
    
    features['image'] = img
    
    return features
