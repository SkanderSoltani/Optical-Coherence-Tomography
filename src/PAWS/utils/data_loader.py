import cv2
# from . import config
import config
import os
import numpy as np

def get_train_test_ds():
    # Mathod to load OCT data set from dir 
    # args:
        # --
    # returns:
    #   X_train,X_test,y_train,y_test as numpy arrays
    def get_data(path):
        # Helper method 
        # Args:
            # path: str representing path to dir
        # Returns:
        #     X,y -> representing dataset and labels
        classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        labels_default  = [1,2,3,4]
        X=[]
        y=[]
        for idx,cl in enumerate(classes):
            label = labels_default [idx]
            dir_path  = os.path.join(path,cl)
            files_list = os.listdir(dir_path)
            for file in files_list:
                file_path = os.path.join(dir_path,file)
                img = cv2.imread(file_path)
                img = cv2.resize(img,(224,224))
                X.append(img)
                y.append(label)

        X = np.array(X) 
        y = np.array(y).reshape((len(X),1))
        return X,y
    X_train,y_train = get_data(config.SOURCE_DS_TRAIN)
    X_test,y_test   = get_data(config.SOURCE_DS_TEST)
    
    return X_train,X_test,y_train,y_test
