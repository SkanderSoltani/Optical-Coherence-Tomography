from keras.preprocessing.image import img_to_array
import operator
import os
from .gradCam import *
from flask import current_app

# last convolutional layer name for GradCam
last_conv_layer_name = 'conv5_block3_out'

# allowed file extensions for the uploaded image
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and process it
    img = image.resize(target)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255

    # return the processed image
    return x


def predict(image, filename, upload_folder):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"predictions": []}
    # define the dictionary of classes
    classes = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
    # preprocess the image and prepare it for classification
    img = prepare_image(image, target=(224, 224))
    # classify the input image and then initialize the list
    # of predictions to return to the client
    preds = current_app.model.predict(img)

    # add all the predictions to the dictionary 'data'
    for idx, prob in enumerate(preds[0]):
        r = {"label": classes[idx], "probability": round(prob, 3)}
        data["predictions"].append(r)

    # get the most likely prediction's label and add to dictionary 'data'
    index, value = max(enumerate(preds[0]), key=operator.itemgetter(1))
    data["max_label"] = []
    data["max_label"].append(classes[index])

    # save gradcam image
    # Remove last layer's softmax
    current_app.model.layers[-1].activation = None

    heatmap = make_gradcam_heatmap(img, current_app.model, last_conv_layer_name)

    cam_path = 'cam_' + filename
    save_and_display_gradcam(image,
                             cam_path=os.path.join(upload_folder, cam_path),
                             heatmap=heatmap)
    # return a dictionary containing all the predictions and the most likely label
    return data, cam_path
