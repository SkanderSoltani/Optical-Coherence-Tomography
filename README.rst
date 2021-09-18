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

.. image:: https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/dataset.png
    :width: 400 px




SimCLR training and evaluation
------------------------------
        SimCLR = a simple framework for contrastive learning of visual representations. It was introduced by Google Research  in 2020. The code is based on this repository: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2 The Simclr folder can be found under src folder.
        SimCLR is a self-supervised method that tries to maximize agreement between two differently augmented views of the same sample image. The base encoder network is Resnet50v2. The model can be trained with SimCLRinTensorflow2.py. The results can be evaluated by adding a linear classifier on top of the base encoder network and training with a small amount of labelled data. This is done in linearEvaluation.py. We were able to reach 96% accuracy with 2% of training data when SimCLR was trained using transfer learning with Resnet50v2 and finetuned with labelled data. Here are the results:
Training:

.. image:: https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/FinetuningSimCLR.png
    :width: 400 px

The confusion matrix and metrics for test data: 
    
.. image:: https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/trainingMetrics.png
    :width: 400 px

t-SNE is used for visualizing the features after the base encoder network:
    
.. image:: https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/featuresTsne.png
    :width: 400 px

Comparison of results
------------------------------

.. image:: https://github.com/annilea/optical-coherence-tomography/blob/master/imgs/FinalResults.png
    :width: 400 px


Web application
---------------
    This application lets the user upload an OCT image and shows classification results along with GradCam generated information about which parts in the image affected the model's decision. The octapp folder contains the files and folders required for a deployable web application to AWS Elastic Beanstalk. The model folder created by Tensorflow's model.save must be called "model" and saved under the folder "static". Also create a folder called "uploads" under static. The user's uploaded images will be saved here along with respective gradcam images. To run the app make sure you have configured AWS credentials. Then install EBS CLI with

    `pip install awsebcli --upgrade --user`

    To deploy, navigate to the octapp folder that contains the application.py. Change the region to whichever you need to use.  Run:

    `eb init -p python-3.7 rpsapp --region us-west-2`

    Then run:

    `eb create rpsapp --instance_type t2.large`

    Finally you can open your application with

    `eb open`

    and close the application with

    `eb terminate octapp`



