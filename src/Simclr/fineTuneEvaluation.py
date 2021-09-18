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

datagen_no_aug_train = data_generator(False)
datagen_no_aug_test = data_generator(False)

# Point the generator directory to the folder were you have the x% split of the train data
train_gen = datagen_no_aug_train.flow_from_directory(directory="../../data/warm_start_data_split/train/",
                                                  target_size=(224,224),
                                                  color_mode="rgb",
                                                  batch_size=32,
                                                  shuffle=False)
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

# Creating a new model using the output of the GlobalAveragePooling2D layer
# 2048 dimension Feature Vector
# The fine_tuned model has the same architecture as our baseline ResNet50V2 supervised model
last_output = pre_trained_resnet_simclr.layers[-6].output
x = Dense(1024, activation='relu')(last_output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0))(x)
#fine_tuned_resnet_simclr = Model(pre_trained_resnet_simclr.input, x)

# Loading the weights or the SavedModel from the fine tune training
#fine_tuned_resnet_simclr.load_weights(filepath="checkPoints/fineTune/2_perc_dataset/20210916-013052")
fine_tuned_resnet_simclr= tf.keras.models.load_model("SimCLR_model/fineTune/2_perc_dataset")

# Initiate a new run on wandb. Change the id parameter for the name of the run you want
wandb.init(project="simclr_oct", entity="gilsanzovo", id="Evaluation-fine_tune-2_perc_dataset")


# Plot evaluation metrics on test data
y_pred = fine_tuned_resnet_simclr.predict(test_gen)
y_true = test_gen.classes
y_pred_max = y_pred.argmax(axis=1)

print(classification_report(y_true=y_true,y_pred=y_pred_max))
print(confusion_matrix(y_true=y_true,y_pred=y_pred_max))


# Visualization of the representations
def plot_vecs_n_labels(v, labels):
    fig = plt.figure(figsize=(10, 10))
    sns.set_style("darkgrid")
    sns.scatterplot(v[:, 0], v[:, 1], hue=labels, legend='full', palette=sns.color_palette("bright", 4))
    plt.show()

    return fig


# Representations with no nonlinear projections
# Representations from the pre-trained SimCLR model
tsne = TSNE()
projection = Model(pre_trained_resnet_simclr.input, pre_trained_resnet_simclr.layers[-6].output)
train_features = projection.predict(train_gen)
print("The feature representation shape from the nonlinear projection of the pre-trained SimCLR model is: ", train_features.shape)
low_vectors = tsne.fit_transform(train_features)
fig = plot_vecs_n_labels(low_vectors, train_gen.classes)
wandb.log({"t-SNE plot for the 2048 feature vector using the pre-trained SimCLR model": fig})

# Representations from the fine-tuned SimCLR model
tsne = TSNE()
projection = Model(fine_tuned_resnet_simclr.input, fine_tuned_resnet_simclr.layers[-5].output)
train_features = projection.predict(train_gen)
print("The feature representation shape from the nonlinear projection of the fined-tuned SimCLR model is: ", train_features.shape)
low_vectors = tsne.fit_transform(train_features)
fig = plot_vecs_n_labels(low_vectors, train_gen.classes)
wandb.log({"t-SNE plot for the 2048 feature vector using the fine-tuned SimCLR model": fig})