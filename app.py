# import the necessary packages
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image
import flask
import os
import io
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'

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
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


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
        prediction = predict(image)
        flask.flash('Image successfully uploaded and displayed below')
        return flask.render_template('upload.html', filename=filename, prediction=prediction)
    else:
        flask.flash('Allowed image types are -> png, jpg, jpeg')
        return flask.redirect(flask.request.url)


def predict(image):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(224, 224))

    # classify the input image and then initialize the list
    # of predictions to return to the client
    preds = model.predict(image)
    results = imagenet_utils.decode_predictions(preds)
    data["predictions"] = []

    # loop over the results and add them to the list of
    # returned predictions
    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": float(prob)}
        data["predictions"].append(r)

    # indicate that the request was a success
    data["success"] = True
    # return the data dictionary as a JSON response
    return data["predictions"]


@app.route('/display/<filename>')
def display_image(filename):
    return flask.redirect(flask.url_for('static', filename='uploads/' + filename), code=301)


# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
