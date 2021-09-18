import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb
import glob
from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
import helpers
import time

# Logging into wandb.ai
wandb.login(key="e215c1e2d81816b1de837c53fa0f9b9a5dbd4407")

# Important variables
train_epochs = 25
BATCH_SIZE = 64
start = time.time()

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)
# following https://github.com/sayakpaul/SimCLR-in-TensorFlow-2
# Train image paths
train_images = glob.glob("../../data/new_data_split/train/*/*")
num_examples = len(train_images)
print("The number of training images is: ", num_examples)

# Augmentation utilities (differs from the original implementation)
# Referred from: https://arxiv.org/pdf/2002.05709.pdf (Appendxi A
# corresponding GitHub: https://github.com/google-research/simclr/)

class CustomAugment(object):
    def __call__(self, sample):
        # cropping
        batch_size = sample.get_shape().as_list()[0]
        #print(batch_size)
        batch_holder = np.zeros((batch_size, 224, 224, 3))
        j=0
        image_shape = 286
        crop_size = 224
        scale = [0.5, 0.9]
        for image in sample:
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

            batch_holder[j, :] = crop_resize
            j = j+1


        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, batch_holder, p=0.5)
        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.2 * s) #changed from 0.8 to avoid black images
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        x = tf.image.random_hue(x, max_delta=0.2 * s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    def _random_apply(self, func, x, p):
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x)


# Build the augmentation pipeline
data_augmentation = Sequential([Lambda(CustomAugment())])
print("added custom augmentations")


# Image preprocessing utils
@tf.function
def parse_images(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    return image



train_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_ds = (
    train_ds
        .map(parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(1024)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
)
print("train_ds done - tf dataset")


# Architecture utils
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = True
    inputs = Input((224, 224, 3))
    h = base_model(inputs, training=True)
    h = GlobalAveragePooling2D()(h)

    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(hidden_3)(projection_2)

    resnet_simclr = Model(inputs, projection_3)

    return resnet_simclr


# Mask to remove positive examples from the batch of negative samples
negative_mask = helpers.get_negative_mask(BATCH_SIZE)
print("created a negative mask")


@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Initiate a new run on wandb. Change the id parameter for the name of the run you want
wandb.init(project="simclr_oct", entity="gilsanzovo", id="SimCLR-full_dataset-25_epochs-training")

def train_simclr(model, dataset, optimizer, criterion,
                 temperature=0.1, epochs=100):
    step_wise_loss = []
    epoch_wise_loss = []
    print("got to training function")
    #print(len(dataset))
    for epoch in tqdm(range(epochs)):
        for image_batch in dataset:
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        wandb.log({"nt_xentloss": np.mean(step_wise_loss),
                   "learning_rate": optimizer._decayed_lr('float32').numpy()})

        if epoch % 5 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))
            model.save_weights("checkPoints/SimCLR-full_dataset-25_epochs-training/sim_weights")
        print("end of epoch. Been running:")
        end = time.time()
        print(end - start)
    print("end of function. Been running:")
    end = time.time()
    print(end - start)
    return epoch_wise_loss, model

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.SUM)
# decay_steps = 1000
decay_steps = num_examples * train_epochs // BATCH_SIZE + 1
print("The number of decay_steps for the learning rate given the number of training images and the batch size is: ", decay_steps)
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

resnet_simclr_2 = get_resnet_simclr(256, 128, 50)
print(f"starting training for {train_epochs} epochs")
epoch_wise_loss, resnet_simclr = train_simclr(resnet_simclr_2, train_ds, optimizer, criterion,
                                              temperature=0.1, epochs=train_epochs)  # epochs=25
print("finished training")
with plt.xkcd():
    plt.plot(epoch_wise_loss)
    plt.title("tau = 0.1, h1 = 256, h2 = 128, h3 = 50")
    plt.show()

resnet_simclr.save_weights("checkPoints/SimCLR-full_dataset-25_epochs-training/sim_weights")
print("saved weights")