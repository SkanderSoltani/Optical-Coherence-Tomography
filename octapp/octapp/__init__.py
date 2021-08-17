# flake8: noqa
from flask import Flask, app
import keras
import os


def create_app():
    """Initialize the core application."""

    app = Flask(__name__)
    UPLOAD_FOLDER = 'octapp/static/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.model = keras.models.load_model(os.path.join('octapp/static', 'model'))
    with app.app_context():
        from . import routes

        return app

