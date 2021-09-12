============================
Optical Coherence Tomography
============================
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


A deep learning system to classify images from Optical Chorerence Tomography (OCT) images


* Free software: MIT license

Data
----
The dataset can be found from: https://data.mendeley.com/datasets/rscbjbr9sj/2
It contains 84,416 images (JPEG) in 4 categories:CNV,DME, DRUSEN, NORMAL. The images are divided to train and test sets.
<img width="446" alt="image" src="https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/dataset.png">

Features
--------
- SimCLR training and evaluation
        SimCLR = a simple framework for contrastive learning of visual representations. It was introduced by Google Research  in 2020. The code is based on this repository: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2 The Simclr folder can be found under src folder.
        SimCLR is a self-supervised method that tries to maximize agreement between two differently augmented views of the same sample image. The base encoder network is Resnet50. The model can be trained with SimCLRinTensorflow2.py. The results can be evaluated by adding a linear classifier on top of the base encoder network. This is done in linearEvaluation.py. The code prints a confusion matrix and metrics and t-SNE is used for visualizing the features after the base encoder network. We were able to reach 94% accuracy with 10% of training data when SimCLR was trained using transfer learning with Resnet50. Here are the results:
        <img width="446" alt="image" src="https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/linearClassifierTraining.png">
        <img width="446" alt="image" src="https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/trainingMetrics.png">
        <img width="446" alt="image" src="https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/featuresTsne.png">

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



