import os
import numpy as np
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import shutil
from random import sample
from datetime import datetime
import cv2
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh

# Model 
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

def get_params():
    # Utility method to load parameters:
    # Args:
        # ---
    # Returns:
        # params -> dictionary with parameters
    with open('config.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return params

#params
params = get_params()

# Optimizer
optim = tf.keras.optimizers.Adam(learning_rate = params['optimizer']['lr'],
                                   beta_1   = params['optimizer']['beta_1'],
                                   beta_2   = params['optimizer']['beta_2'],
                                   epsilon  = params['optimizer']['epsilon'],
                                   amsgrad  = params['optimizer']['amsgrad'])




params= get_params()
def create_model(params):
    # function to create keras model
    # Args:
    #     params -> dict representing config params
    # Returns:
    #     model -> Keras model 
    pretrained_model = tf.keras.applications.ResNet50V2(include_top=True,weights="imagenet")
    
    last_layer  = pretrained_model.get_layer('avg_pool')
    last_output = last_layer.output
    x = layers.Dense(1024,activation='relu')(last_output)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dense(4,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(l2=0.0))(x)
    model = Model(pretrained_model.input,x)
    return model

def get_callbacks(call_backs_dir):
    # Method to create callbacks used in keras
    # Args:
    #   call_backs_dir -> str representing directory path to call backs
    # Returns:
    #   call_backs -> list() representing the list of callbacks

    # tensorboard -log files 
    log_dir = os.path.join(params['data']['path_to_logs'],call_backs_dir,datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # checkpoints to save model
    checkpoint_dir = os.path.join(params['data']['path_to_checkpoints'],call_backs_dir,'checkpoints')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath =checkpoint_dir,
                                                         save_freq = 'epoch',
                                                         save_weights_only=True,
                                                         monitor='val_accuracy',
                                                         save_best_only=True)
    # forming list of callbacks
    call_backs = [tensorboard_callback,checkpoint_callback]
    return call_backs


def evaluate(y_true,y_pred):
        # Method to evaluate predictions using classification_report and confusion_matrix from sklearn
        # Args:
            # y_true -> numpy.array(int) representing the ground truth sentiment
            # y_pred -> numpy.array(int) representing the predicted sentiment
        # Returns:
            # prints classification report and confusion matrix
        print(classification_report(y_true=y_true,y_pred=y_pred))
        print(confusion_matrix(y_true=y_true,y_pred=y_pred))
    
