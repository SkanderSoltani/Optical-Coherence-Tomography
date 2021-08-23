"""
References:
    * https://arxiv.org/pdf/2104.13963.pdf
    * https://github.com/szacho/augmix-tf
    * https://github.com/ayulockin/SwAV-TF/blob/master/initial_notebooks/Building_MultiCropDataset.ipynb
"""

import tensorflow_addons as tfa
import tensorflow as tf


SIZE_CROPS = [224, 224]  # 32: global views, 18: local views
NUM_CROPS = [2, 6]
GLOBAL_SCALE = [0.75, 1.0]
LOCAL_SCALE = [0.3, 0.75]
AUTO = tf.data.AUTOTUNE


@tf.function
def float_parameter(level, maxval):
    """
    Type-casting to float.

    :param level: determines the intensity of a transformation (should
    be in [1, 10] range) (int)
    :param maxval: upper limit of the pixel (int)
    :return: updated upper limit casted to float
    """
    return tf.cast(level * maxval / 10.0, tf.float32)


@tf.function
def sample_level(n):
    """
    Uniform sampling with [0.1, n) range.
    """
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)


@tf.function
def solarize(image, level=6):
    """
    Performs solarization.

    :param image: input image (h, w, nb_channels)
    :param level: intensity level (int)
    :return: solarized image
    """
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 255 - image)


@tf.function
def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    """
    Randomly applies color distortion.

    :param x: input image (h x w x nb_channels)
    :param strength: strength of distortions
    :return: distorted image
    """
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    x = tf.clip_by_value(x, 0, 255)
    return x


@tf.function
def random_apply(func, x, p):
    """
    Utility to apply a function to a given input with a probability.
    """
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


@tf.function
def random_resize_distort_crop(image, scale, crop_size):
    """
    Applies random-resized cropping and other augmentations including
    flipping, color distortions (includes solarization and equalization).

    :param image: input image (h, w, nb_channels)
    :param scale: scale to be used during random-resized crop (ex - [0.3, 0.7])
    :param crop_size: size to crop to (int)
    :return: random-resized and distorted image
    """
    # Conditional resizing
    if crop_size == 224:
        image_shape = 286
        image = tf.image.resize(image, (image_shape, image_shape))
    else:
        image_shape = 286
        image = tf.image.resize(image, (image_shape, image_shape))
    # Get the crop size for given scale
    size = tf.random.uniform(
        shape=(1,),
        minval=scale[0] * image_shape,
        maxval=scale[1] * image_shape,
        dtype=tf.float32,
    )
    size = tf.cast(size, tf.int32)[0]
    # Get the crop from the image
    crop = tf.image.random_crop(image, (size, size, 3))
    crop_resize = tf.image.resize(crop, (crop_size, crop_size))

    # Flip and color distortions
    image = tf.image.random_flip_left_right(crop_resize)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(solarize, image, p=0.8)
    image = random_apply(tfa.image.equalize, image, p=0.8)
    return image


def get_multicrop_loader(ds: tf.data.Dataset):
    """
    Returns a multi-crop dataset.

    :param ds: a TensorFlow dataset object (containing only unlabeled images)
    :return: a multi-crop dataset
    """
    loaders = tuple()
    for i, num_crop in enumerate(NUM_CROPS):
        for _ in range(num_crop):
            if SIZE_CROPS[i] == 224:
                scale = GLOBAL_SCALE
            elif SIZE_CROPS[i] == 224:
                scale = LOCAL_SCALE

            loader = ds.map(
                lambda x: random_resize_distort_crop(x, scale, SIZE_CROPS[i]),
                num_parallel_calls=AUTO,
                deterministic=True,
            )
            loaders += (loader,)

    return tf.data.Dataset.zip(loaders)
