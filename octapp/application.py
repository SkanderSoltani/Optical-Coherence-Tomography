from octapp import create_app

# Create the Flask app
application = app = create_app()

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    app.run()
