import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from .metric_ds_mapper import MetricDatasetMapper


acc_metric = MetricDatasetMapper(tf.keras.metrics.Accuracy())


def plot_confusion(model, dataset, 
                   class_names=None, 
                   title=None,
                   ax=None,
                   x_tick_rotation=45, 
                   y_tick_rotation=45,
                   **sns_kwargs):
    if not class_names:
        # assume cifar-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataset:
        predictions = model(inputs)
        _, n_classes = tf.shape(predictions)
        class_preds = tf.slice(predictions, [0, 0], [-1, n_classes])

        all_preds.append(class_preds.numpy())
        all_labels.append(labels)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    n = min(preds.shape[0], labels.shape[0])

    cnf_matrix = confusion_matrix(np.argmax(labels, axis=-1)[:n], 
                                  np.argmax(preds, axis=-1)[:n],
                                  normalize='true')
    
    no_ax_provided = not ax
    if no_ax_provided:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(cnf_matrix, square=True, ax=ax, 
                vmin=0, vmax=1, **sns_kwargs)
    ax.set_xticks([x for x in range(n_classes)])
    ax.set_yticks([y for y in range(n_classes)])
    ax.set_yticklabels(class_names, rotation=y_tick_rotation)
    ax.set_xticklabels(class_names, rotation=x_tick_rotation)
    if title:
        ax.set_title(title)
    if no_ax_provided:
        plt.show()
    
    
def visualise_predictions(model, dataset, shuffle=False, 
                          class_names=None):
    if not class_names:
        # assume cifar-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    if shuffle:
        dataset = dataset.shuffle()
    test_batch = list(dataset.take(1)).pop()
    test_inputs, test_targets = test_batch

    outputs = model(test_inputs)
    class_preds = tf.slice(outputs, [0, 0], [-1, 10])

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_inputs.numpy()[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index

        class_pred = class_names[tf.argmax(class_preds, axis=-1).numpy()[i]]
        class_true = class_names[tf.argmax(test_targets, axis=-1).numpy()[i]]
        plt.xlabel(f'{class_pred} ({class_true})')
    plt.show()
