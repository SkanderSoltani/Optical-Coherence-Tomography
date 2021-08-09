# -*- coding: utf-8 -*-

import os
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import copy

# tf
import tensorflow as tf

#keras
import keras
import keras.backend as K
from keras import regularizers, optimizers
from keras.layers import Dense
from keras.initializers import Constant

from keras import Model 


from sklearn.metrics import classification_report, confusion_matrix

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_data(path):
	return os.path.join(_ROOT, 'data', path)


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


#############################################################################
#
# 	              OCT Classifier class
#
#############################################################################

class OCT_classifier(Model):
    # child class inherited from "keras.Model" defining the sentiment classifier model
    # methods:
    #   call     -> Method defining the forward pass in the classifier model
    #   evaluate -> Method defining model evaluation metrics a.k.a confusion matrix, precision, recall, ROC,...etc. 
    
    def __init__(self):
        super(OCT_classifier, self).__init__()
        params = get_params()
        # path parameters
        self._path_to_logs         = params['data']['path_to_logs']
        self._path_to_checkpoints  = params['data']['path_to_checkpoints']
        self._path_to_model        = params['data']['path_to_model']
        
        
        # Model compile
        self._model_loss      = params['compile']['loss']
        self._model_metrics   = params['compile']['metrics']
        
        # Model optimizer
        self._lr            = params['optimizer']['lr']
        self._beta_1        = params['optimizer']['beta_1']
        self._beta_2        = params['optimizer']['beta_2']
        self._epsilon       = params['optimizer']['epsilon']
        self._amsgrad       = params['optimizer']['amsgrad']
        self._opt = keras.optimizers.Adam(learning_rate=self._lr,
                                   beta_1 =self._beta_1,
                                   beta_2=self._beta_2,
                                   epsilon=self._epsilon,
                                   amsgrad=self._amsgrad)
        
        
        # Model Fit
        self._batch_size            = params['fit']['batch_size']
        self._epochs                = params['fit']['epochs']
        self._verbose               = params['fit']['verbose']
        self._validation_split      = params['fit']['validation_split']
        self._shuffle               = params['fit']['shuffle']
        self._class_weight          = params['fit']['class_weight']
        self._sample_weight         = params['fit']['sample_weight']
        self._initial_epoch         = params['fit']['initial_epoch']
        self._steps_per_epoch       = params['fit']['steps_per_epoch']
        self._validation_steps      = params['fit']['validation_steps']
        self._validation_batch_size = params['fit']['validation_batch_size']
        self._validation_freq       = params['fit']['validation_freq']
        self._max_queue_size        = params['fit']['max_queue_size']
        self._workers               = params['fit']['workers']
        self._use_multiprocessing   = params['fit']['use_multiprocessing']
        
        # Model Predict:
        self._batch_size_p          = params['predict']['batch_size']
        self._verbose_p             = params['predict']['verbose']
        self._steps_p               = params['predict']['steps'] 
        self._max_queue_size_p      = params['predict']['max_queue_size']
        self._workers_p             = params['predict']['workers']
        self._use_multiprocessing_p = params['predict']['use_multiprocessing']
        
                 ###############################
                 ####### MODEL STRUCTURE #######
                 ###############################
       
        
        # Model structure
        self.pretrained_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                                 weights="imagenet",
                                                                 input_shape=(224,224,3))

            
    
        self.last_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")
        self.dense1024     = tf.keras.layers.Dense(1024,activation='relu')
        self.dense512      = tf.keras.layers.Dense(512,activation='relu')
        self.dense256      = tf.keras.layers.Dense(256,activation='relu')
        self.dense4        = tf.keras.layers.Dense(4,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(l2=0.0))
        
    
    def call(self,inputs):
        # Method defining the forward pass in the classifier model (called internally during forward pass in compile method)
        # Args:
        #   inputs     ->  np.array() representing tokens 
        # returns:
        #   lastTensor -> tensor representing last layer in the model    
        
        x = self.pretrained_model(inputs)
        x = self.last_avg_pool(x)
        x = self.dense1024(x)
        x = self.dense512(x)
        x = self.dense256(x) 
        lastTensor = self.dense4(x)
        return lastTensor
    
    def Compile(self):
        # Method to compile the model
        self.compile(optimizer=self._opt, 
                     loss=self._model_loss, 
                     metrics=self._model_metrics)
        
    def Fit(self,training_gen,validation_gen=None,call_backs_dir = None):
        # Method to fit the model to current data
        # Args:
        #    training_data   -> ImageDataGenerator Object Flow with training data
        #    validation_data -> ImageDataGenerator Object Flow with validation data
        #    call_back_dir   -> optional, of type str describing name of log directory for callbacks, default is None
        # Returns:
        #    ---
        if call_backs_dir:
            # tensorboard -log files 
            log_dir = os.path.join(self._path_to_logs,call_backs_dir,datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
            
            # checkpoints to save model
            checkpoint_dir = os.path.join(self._path_to_checkpoints,call_backs_dir,'')
            
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath =checkpoint_dir,
                                                                     save_freq = 'epoch',
                                                                     save_weights_only=True,
                                                                     monitor='val_accuracy',
                                                                     save_best_only=True)
            # forming list of callbacks
            call_backs = [tensorboard_callback,checkpoint_callback]
        else:
            call_backs = None
            
            
        
        self.fit(x                 = training_gen,
            #  batch_size            = self._batch_size, 
             epochs                = self._epochs, 
             verbose               = self._verbose,
             callbacks             = call_backs, 
             validation_split      = self._validation_split, 
             validation_data       = validation_gen,
             shuffle               = self._shuffle,
             class_weight          = self._class_weight, 
             sample_weight         = self._sample_weight, 
             initial_epoch         = self._initial_epoch, 
             steps_per_epoch       = self._steps_per_epoch,
             validation_steps      = self._validation_steps, 
             validation_batch_size = self._validation_batch_size,
             validation_freq       = self._validation_freq,
             max_queue_size        = self._max_queue_size, 
             workers               = self._workers)
        
    def Predict(self,test_gen,thresh = None):
        # Method to predict sentiment around text
        # Args:
            # test_gen -> ImageDataGenerator Flow object representing test flow
            # thresh   -> of type float. threshold used around predicted probabilities
        # returns:
            # y_proba -> numpy.array() representing vector of probabilities for each class
            # y_pred  -> numpy.array() representing vector of predicted classes
        y_proba = self.predict(test_gen,
                    # batch_size          = self._batch_size_p ,
                    verbose             = self._verbose_p,
                    steps               = self._steps_p,
                    max_queue_size      = self._max_queue_size_p,
                    workers             = self._workers_p,
                    use_multiprocessing = self._use_multiprocessing_p)
        
        y_pred = y_proba.argmax(axis=1)
        return y_proba,y_pred
    
    
    def Evaluate(self,y_true,y_pred):
        # Method to evaluate predictions using classification_report and confusion_matrix from sklearn
        # Args:
            # y_true -> numpy.array(int) representing the ground truth sentiment
            # y_pred -> numpy.array(int) representing the predicted sentiment
        # Returns:
            # prints classification report and confusion matrix
        print(classification_report(y_true=y_true,y_pred=y_pred))
        print(confusion_matrix(y_true=y_true,y_pred=y_pred))
        
    def Load_weights(self,checkpoint_dir):
        # Method to load weights from checkpoints - often used during traing in conjuction with tensorboard to monitor
        # training performance
        try:
            path = os.path.join(self._path_to_checkpoints,checkpoint_dir,'/')
            self.load_weights(path).expect_partial()
            print("Weights Loaded") # This should be a sent to a log file down the road
        except:
            print("No checkpoints saved!")
    
    def Save(self,version_name):
        # Method to save model to " data\\saved_models\\version"
        # Args:
            # version_dir -> of type str() representing directory name aka: V1, V2, V3...etc
        # Returns:
            # --- prints / logs a statement of success or failure
        
        path_to_model = os.path.join(self._path_to_model,version_name)
        try:
            self.save(path_to_model)
            print("model saved successfully!")
        except:
            print("Warning: model was not saved!")
    
    def Load_model(self,version_name):
        # Method to load model from " data\\saved_models\\version"
        # Args:
            # version_dir -> of type str() representing directory name aka: V1, V2, V3...etc
        # Returns:
            # --- prints / logs a statement of success or failure
        path_to_model = os.path.join(self._path_to_model,version_name)
        try:
            reconstructed_model = keras.models.load_model(path_to_model)
            print("model loaded successfully!")
            return reconstructed_model
        except:
            print("Warning: model was not loaded!")