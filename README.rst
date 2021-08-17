============================
Optical Coherence Tomography
============================


.. image:: https://img.shields.io/travis/SkanderSoltani/potholes_detection_system.svg
        :target: https://travis-ci.org/SkanderSoltani/potholes_detection_system

.. image:: https://readthedocs.org/projects/potholes-detection-system/badge/?version=latest
        :target: https://potholes-detection-system.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


A deep learning system to detect roads cracks and potholes


* Free software: MIT license


Features
--------
- Web application
    This application lets the user upload an OCT image and shows classification results along with GradCam generated information about which parts in the image affected the model's decision. The octapp folder contains the files and folders required for a deployable web application to AWS Elastic Beanstalk. The model folder created by Tensorflow's model.save must be called "model" and saved under the folder "static". Also create a folder called "uploads" under static. The user's uploaded images will be saved here along with respective gradcam images. To run the app make sure you have configured AWS credentials. Then install EBS CLI with

    `pip install awsebcli --upgrade --user`

    To deploy, navigate to the octapp folder that contains the application.py. Run

    `eb init -p python-3.7 rpsapp --region us-west-2`

    Change the region to whichever you need to use. Then run:

    `eb create rpsapp --instance_type t2.large`

    Finally you can open your application with

    `eb open`

    and close the application with

    `eb terminate octapp`



