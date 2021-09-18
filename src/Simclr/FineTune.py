from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import glob
import wandb
from wandb.keras import WandbCallback
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Logging into wandb.ai
wandb.login(key="e215c1e2d81816b1de837c53fa0f9b9a5dbd4407")

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

# Function to create the ImageDataGenerator
# Using the same parameters we used to train the Supervised Model
def data_generator(aug=True):
    if aug:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     featurewise_center=False,
                                     samplewise_center=False,
                                     featurewise_std_normalization=False,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     zca_epsilon=0.0000000001,
                                     rotation_range=45,
                                     width_shift_range=0.0,
                                     height_shift_range=0.0,
                                     brightness_range=[0,0.8],
                                     shear_range=0.0,
                                     zoom_range=0.0,
                                     channel_shift_range=0.0,
                                     fill_mode="nearest",
                                     cval=0.0,
                                     horizontal_flip=True,
                                     vertical_flip=True)
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen

datagen_aug_train = data_generator(True)
datagen_no_aug_val = data_generator(False)
datagen_no_aug_test = data_generator(False)

#Create the Generators for the Train, Val and Test sets using Flow from Directory
# Point the generator directory to the folder were you have the x% split of the train data
train_gen = datagen_aug_train.flow_from_directory(directory="../../data/warm_start_data_split/train/",
                                                  target_size=(224,224),
                                                  color_mode="rgb",
                                                  batch_size=32,
                                                  shuffle=True)

# Point the generator directory to the folder were you have the x% split of the val data
val_gen = datagen_no_aug_val.flow_from_directory(directory="../../data/warm_start_data_split/val/",
                                                  target_size=(224,224),
                                                  color_mode="rgb",
                                                  batch_size=16,
                                                  shuffle=True)

# Point the generator directory to the folder were you have the test data
test_gen = datagen_no_aug_test.flow_from_directory(directory="../../data/OCT2017/test/",
                                                   target_size=(224, 224),
                                                   color_mode="rgb",
                                                   batch_size=32,
                                                   shuffle=False)


# Function to build a model with the same architecture used to pretrain the SimCLR model
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(224, 224, 3))
    base_model.trainable = True
    inputs = Input((224, 224, 3))
    h = base_model(inputs, training=False)
    h = GlobalAveragePooling2D()(h)

    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(hidden_3)(projection_2)

    resnet_simclr = Model(inputs, projection_3)

    return resnet_simclr

# Loading the pretrained SimCLR model
pre_trained_resnet_simclr = get_resnet_simclr(256, 128, 50)
pre_trained_resnet_simclr.load_weights(filepath="checkPoints/v7-full_dataset-100_epochs/sim_weights")
pre_trained_resnet_simclr.summary()

# Creating a new model using the output of the GlobalAveragePooling2D layer
# 2048 dimension Feature Vector
# The fine_tuned model has the same architecture as our baseline ResNet50V2 supervised model
last_output = pre_trained_resnet_simclr.layers[-6].output
x = Dense(1024, activation='relu')(last_output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0))(x)
fine_tuned_resnet_simclr = Model(pre_trained_resnet_simclr.input, x)
fine_tuned_resnet_simclr.summary()


#Function to plot the training loss and accuracy per epoch
def plot_training(H):
    with plt.xkcd():
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()

# Initiate a new run on wandb. Change the id parameter for the name of the run you want
wandb.init(project="simclr_oct", entity="gilsanzovo", id="mac-fine_tune-5_perc_dataset")


# checkpoints to save model
checkpoint_dir = os.path.join('checkPoints', 'fineTune/5_perc_dataset', datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                         save_freq='epoch',
                                                         save_weights_only=True,
                                                         monitor='val_accuracy',
                                                         save_best_only=True)
# forming list of callbacks - checkpoints and wandb.ai
call_backs = [checkpoint_callback, WandbCallback()]


fine_tuned_resnet_simclr.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

history = fine_tuned_resnet_simclr.fit(train_gen,
                           validation_data=val_gen,
                           epochs=80,
                           callbacks=call_backs)

plot_training(history)
# Plot evaluation metrics on test data
y_pred = fine_tuned_resnet_simclr.predict(test_gen)
y_true = test_gen.classes
y_pred_max = y_pred.argmax(axis=1)

print(classification_report(y_true=y_true,y_pred=y_pred_max))
print(confusion_matrix(y_true=y_true,y_pred=y_pred_max))
fine_tuned_resnet_simclr.save("SimCLR_model/fineTune/5_perc_dataset")