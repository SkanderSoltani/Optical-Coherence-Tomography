# Imports
from utils import (
    multicrop_loader,
    labeled_loader,
    paws_trainer,
    config,
    lr_scheduler,
    lars_optimizer,
)
from models import wide_resnet
import matplotlib.pyplot as plt
import tensorflow as tf
import time

#Original:
# Load dataset
#(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

#Our data load:
from tqdm import tqdm
import os
import cv2
import numpy as np
from skimage.transform import resize


imageSize = 224
train_dir = "../../data/warm_start_data_split/train/"
#test_dir = "../../data/warm_start_data_split/val/"


# ['DME', 'CNV', 'NORMAL', '.DS_Store', 'DRUSEN']
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['CNV']:
                label = 1
            elif folderName in ['DME']:
                label = 2
            elif folderName in ['DRUSEN']:
                label = 3
            else:
                label = 4

            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


x_train, y_train = get_data(train_dir)
#x_test, y_test = get_data(test_dir)

# Constants
AUTO = tf.data.AUTOTUNE
STEPS_PER_EPOCH = int(len(x_train) // config.MULTICROP_BS)
WARMUP_EPOCHS = 10
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

# Prepare Dataset object for multicrop
train_ds = tf.data.Dataset.from_tensor_slices(x_train)
multicrop_ds = multicrop_loader.get_multicrop_loader(train_ds)
multicrop_ds = (
    multicrop_ds.shuffle(config.MULTICROP_BS * 10)
    .batch(config.MULTICROP_BS)
    .prefetch(AUTO)
)

# Prepare dataset object for the support samples
support_ds = labeled_loader.get_support_ds(config.SUPPORT_BS)
print("Data loaders prepared.")

# Initialize encoder and optimizer
wide_resnet_enc = wide_resnet.get_network()
scheduled_lrs = lr_scheduler.WarmUpCosine(
    learning_rate_base=config.WARMUP_LR,
    total_steps=config.PRETRAINING_EPOCHS * STEPS_PER_EPOCH,
    warmup_learning_rate=config.START_LR,
    warmup_steps=WARMUP_STEPS,
)
optimizer = lars_optimizer.LARS(
    learning_rate=scheduled_lrs,
    momentum=0.9,
    exclude_from_weight_decay=["batch_normalization", "bias"],
    exclude_from_layer_adaptation=["batch_normalization", "bias"],
)
print("Model and optimizer initialized.")

# Loss trackers
epoch_ce_losses = []
epoch_me_losses = []

############## Training ##############
for e in range(config.PRETRAINING_EPOCHS):
    print(f"=======Starting epoch: {e}=======")
    start_time = time.time()
    epoch_ce_loss_avg = tf.keras.metrics.Mean()
    epoch_me_loss_avg = tf.keras.metrics.Mean()

    for i, unsup_imgs in enumerate(multicrop_ds):
        # Sample support images, concat the images and labels, and
        # then apply label-smoothing.
        global iter_supervised
        try:
            sdata = next(iter_supervised)
        except Exception:
            iter_supervised = iter(support_ds)
            sdata = next(iter_supervised)
        support_images_one, support_images_two = sdata
        support_images = tf.concat(
            [support_images_one[0], support_images_two[0]], axis=0
        )
        support_labels = tf.concat(
            [support_images_one[1], support_images_two[1]], axis=0
        )
        support_labels = labeled_loader.onehot_encode(
            support_labels, config.LABEL_SMOOTHING
        )

        # Perform training step
        batch_ce_loss, batch_me_loss, gradients = paws_trainer.train_step(
            unsup_imgs, (support_images, support_labels), wide_resnet_enc
        )

        # Update the parameters of the encoder
        optimizer.apply_gradients(zip(gradients, wide_resnet_enc.trainable_variables))

        if (i % 50) == 0:
            print(
                "[%d, %5d] loss: %.3f (%.3f %.3f)"
                % (
                    e,
                    i,
                    batch_ce_loss.numpy() + batch_me_loss.numpy(),
                    batch_ce_loss.numpy(),
                    batch_me_loss.numpy(),
                )
            )
        epoch_ce_loss_avg.update_state(batch_ce_loss)
        epoch_me_loss_avg.update_state(batch_me_loss)

    print(
        f"Epoch: {e} CE Loss: {epoch_ce_loss_avg.result():.3f}"
        f" ME-MAX Loss: {epoch_me_loss_avg.result():.3f}"
        f" Time elapsed: {time.time()-start_time:.2f} secs"
    )
    print("")
    epoch_ce_losses.append(epoch_ce_loss_avg.result())
    epoch_me_losses.append(epoch_me_loss_avg.result())

# Create a plot to see the cross-entropy losses
plt.figure(figsize=(8, 8))
plt.plot(epoch_ce_losses)
plt.title("PAWS Training Cross-Entropy Loss", fontsize=12)
plt.grid()
plt.savefig(config.PRETRAINING_PLOT, dpi=300)

# Serialize model
wide_resnet_enc.save(config.PRETRAINED_MODEL)
print(f"Encoder serialized to : {config.PRETRAINED_MODEL}")
