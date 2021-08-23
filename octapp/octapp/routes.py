from flask import current_app as app
from .predict import *
from werkzeug.utils import secure_filename
from PIL import Image
import flask
import os
from tensorflow import keras
from flask import current_app as app
# path to the folder where the user's uploaded images are saved
UPLOAD_FOLDER = os.path.join('static','uploads')


@app.route("/")
def upload_form():
    return flask.render_template('classify.html')


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
        prediction, cam_path = predict(image, filename, app.config['UPLOAD_FOLDER'])
        # flask.flash('Image successfully uploaded and displayed below with predicted class')
        return flask.render_template('classify.html', filename=filename, prediction=prediction, cam_path=cam_path)
    else:
        flask.flash('Allowed image types are -> png, jpg, jpeg')
        return flask.redirect(flask.request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return flask.redirect(flask.url_for('static', filename='uploads/' + filename), code=301)
