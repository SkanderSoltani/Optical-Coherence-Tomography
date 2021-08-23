from . import multicrop_loader, config
import tensorflow as tf
import numpy as np
import os
from . import data_loader

GLOBAL_SCALE = [0.75, 1.0]
AUTO = tf.data.AUTOTUNE
# (X_TRAIN, Y_TRAIN), (_, _) = tf.keras.datasets.cifar10.load_data()
X_TRAIN,_,Y_TRAIN,_ = data_loader.get_train_test_ds()


def onehot_encode(labels, label_smoothing=0.1):
    """
    One-hot encode labels with label smoothing.

    :param labels: (batch_size, )
    return: one-hot encoded labels with optional label smoothing
    """
    labels = tf.one_hot(labels, depth=4)
    # Reference: https://t.ly/CSYO)
    labels *= 1.0 - label_smoothing
    labels += label_smoothing / labels.shape[1]
    return labels


def sample_dataset():
    """
    Returns a randomly sampled dataset and saves the randomly sampled
    indices.
    """
    if not os.path.exists(config.SUPPORT_IDX):
        sampled_idx = np.random.choice(len(X_TRAIN), config.SUPPORT_SAMPLES)
        np.save(config.SUPPORT_IDX, sampled_idx)
        print(1)
    else:
        sampled_idx = np.load(config.SUPPORT_IDX)

    sampled_train, sampled_labels = X_TRAIN[sampled_idx], Y_TRAIN[sampled_idx].squeeze()
    sampled_labels = tf.cast(sampled_labels,tf.int64)
    return tf.data.Dataset.from_tensor_slices((sampled_train, sampled_labels))


def dataset_for_class(i):
    """
    Returns a dataset containing filtered with given class label.

    :param ds: TensorFlow Dataset object
    :param i: class label
    :return: filtered dataset
    """
    ds = sample_dataset()
    i = tf.cast(i, tf.uint8)
    return ds.filter(lambda image, label: label == i).repeat()


def get_support_ds(bs, aug=True):
    """
    Prepares TensorFlow dataset with sampling as suggested in:
    https://arxiv.org/abs/2104.13963 (See Appendix C)

    :param bs: batch size (int)
    :param aug: boolean indicating if augmentation should be applied
    :return: a multi-crop dataset
    """
    # As per Appendix C, for CIFAR10 2x views are needed for making
    # the network better at instance discrimination.
    # Reference:
    # https://stackoverflow.com/questions/46938530/
    ds = tf.data.Dataset.range(10).interleave(
        dataset_for_class,
        cycle_length=10,
        num_parallel_calls=AUTO,
        deterministic=True,
    )

    # As per Appendix C, for CIFAR10 2x views are needed for making
    # the network better at instance discrimination.
    loaders = tuple()
    for _ in range(config.SUP_VIEWS):
        if aug:
            loader = ds.map(
                lambda x, y: (
                    multicrop_loader.random_resize_distort_crop(x, GLOBAL_SCALE, 224),
                    y,
                ),
                num_parallel_calls=AUTO,
                deterministic=True,
            )
        else:
            loader = ds
        loaders += (loader,)

    final_ds = tf.data.Dataset.zip(loaders)
    return final_ds.batch(bs).prefetch(AUTO)
