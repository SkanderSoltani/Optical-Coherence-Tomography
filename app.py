# import the necessary packages
import gradCam
from keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import operator
import flask
import os
import keras

# path to the folder where the user's uploaded images are saved
UPLOAD_FOLDER = 'static/uploads'
# last convolutional layer name for GradCam
last_conv_layer_name = 'conv5_block3_out'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

app.secret_key = "tosiuniikkisalainenavain"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    # load the model
    global model
    model = keras.models.load_model('src/OCT_Model/logs/saved_models/Vt/')


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and process it
    img = image.resize(target)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255

    # return the processed image
    return x


@app.route("/")
def upload_form():
    return flask.render_template('upload.html')


@app.route("/", methods=["POST"])
def upload_image():
    if 'file' not in flask.request.files:
        flask.flash('No file part')
        return flask.redirect(flask.request.url)
    file = flask.request.files['file']
    if file.filename == '':
        flask.flash('No image selected for uploading')
        return flask.redirect(flask.request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = Image.open(file)
        prediction, cam_path = predict(image, filename)
        # flask.flash('Image successfully uploaded and displayed below with predicted class')
        return flask.render_template('upload.html', filename=filename, prediction=prediction, cam_path=cam_path)
    else:
        flask.flash('Allowed image types are -> png, jpg, jpeg')
        return flask.redirect(flask.request.url)


def predict(image, filename):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"predictions": []}
    # define the dictionary of classes
    classes = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}

    # preprocess the image and prepare it for classification
    img = prepare_image(image, target=(224, 224))

    # classify the input image and then initialize the list
    # of predictions to return to the client
    preds = model.predict(img)

    # add all the predictions to the dictionary 'data'
    for idx, prob in enumerate(preds[0]):
        r = {"label": classes[idx], "probability": round(prob, 3)}
        data["predictions"].append(r)

    # get the most likely prediction's label and add to dictionary 'data'
    index, value = max(enumerate(preds[0]), key=operator.itemgetter(1))
    data["max_label"] =[]
    data["max_label"].append(classes[index])

    # save gradcam image
    # Remove last layer's softmax
    model.layers[-1].activation = None

    heatmap = gradCam.make_gradcam_heatmap(img, model, last_conv_layer_name)

    cam_path = 'cam_' + filename
    gradCam.save_and_display_gradcam(image,cam_path=os.path.join(app.config['UPLOAD_FOLDER'], cam_path), heatmap=heatmap, disp=False)
    # return a dictionary containing all the predictions and the most likely label
    return data, cam_path


@app.route('/display/<filename>')
def display_image(filename):
    return flask.redirect(flask.url_for('static', filename='uploads/' + filename), code=301)


# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host='127.0.0.1', port=5000)
