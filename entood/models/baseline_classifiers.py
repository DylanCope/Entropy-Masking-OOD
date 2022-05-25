import tensorflow as tf
from tensorflow.keras import datasets, layers, models


data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2, fill_mode='nearest'),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1, fill_mode='nearest'),
])


class ClassifierOnAE(tf.keras.Model):
    
    def __init__(self,
                 pretrained_ae,
                 name='classifier',
                 num_classes=10,
                 dropout=0.2,
                 train_ae=False):
        
        super(ClassifierOnAE, self).__init__(name=name)
        
        self.classifier = tf.keras.Sequential([
            data_augmentation,
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(512, 
                                  activation='relu',
                                  name='classifier_hidden_1'),
            tf.keras.layers.Dense(256, 
                                  activation='relu',
                                  name='classifier_hidden_2'),
            tf.keras.layers.Dense(128, 
                                  activation='relu',
                                  name='classifier_hidden_3'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, 
                                  activation='relu',
                                  name='classifier_hidden_4'),
            tf.keras.layers.Dense(num_classes, 
                                  activation=None,
                                  name='classifier_out'),
        ], name='classifier')
        
    def call(self, inputs, training=False):
        return self.classifier(inputs, training=training)
    

class ClassifierFineTuneAE(tf.keras.Model):
    
    def __init__(self,
                 pretrained_ae,
                 name='classifier_ft_ae',
                 num_classes=10,
                 dropout=0.2,
                 data_aug=None,
                 train_ae=False):
        
        super(ClassifierFineTuneAE, self).__init__(name=name)
        self.data_augmentation = data_aug or data_augmentation
        
        self.ae = pretrained_ae
        
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256, 
                                  activation='relu',
                                  name='classifier_hidden_1'),
            tf.keras.layers.Dense(64, 
                                  activation='relu',
                                  name='classifier_hidden_2'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, 
                                  activation='relu',
                                  name='classifier_hidden_3'),
            tf.keras.layers.Dense(num_classes, 
                                  activation=None,
                                  name='classifier_out'),
        ], name='classifier')
        
    @property
    def trainable_variables(self):
        return self.classifier.trainable_variables + self.ae.encoder.trainable_variables
        
    def call(self, inputs, training=False):
        inputs = self.data_augmentation(inputs, training=training)
        enc = self.ae.encode(inputs, training=training)
        return self.classifier(enc, training=training)
    
    
class ClassifierNoAE(tf.keras.Model):
    
    def __init__(self,
                 name='classifier',
                 num_classes=10,
                 dropout=0.2,
                 train_ae=False):
        
        super(ClassifierNoAE, self).__init__(name=name)
        
        self.classifier = tf.keras.Sequential([
            data_augmentation,
            tf.keras.layers.Conv2D(32, (3, 3), 
                                   activation='relu', 
                                   input_shape=(32, 32, 3), 
                                   ),#strides=2),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), 
                                   activation='relu'),#, strides=2),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, 
                                  activation='relu',
                                  name='classifier_hidden_1'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, 
                                  activation='relu',
                                  name='classifier_hidden_2'),
            tf.keras.layers.Dense(num_classes, 
                                  activation=None,
                                  name='classifier_out'),
        ], name='classifier')

    def call(self, inputs, training=False):
        return self.classifier(inputs, training=training)
