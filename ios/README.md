# OCT-ai: An iOS app to classify retinal oct images on the fly

## Demo video

https://user-images.githubusercontent.com/101695/159761080-6c7bde2b-15b9-4c94-b58f-c8e994a7e97e.mp4


## Overview

This app is an adaptation of the [TensorFlow Lite iOS image classification example](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/) developed by the Tensorflow team. We have adapted their code to use our ResNet50-V2 tflite model that was trained on retinal oct images. For more information about the inner works of the iOS app itself, please reference the original application on Tensorflow's Github (link above).

The app continuously classify whatever it sees from the device's back camera, using a quantized ResNet50-V2 model. It also works with our quantized MobileNetV3 model but we chose to use the ResNet50-V2 model because its quantized version has considerable better classification performance than the MobileNet, although it's more demanding on the processor and bigger in size. We tested the app on an iPhone X running iOS 15 and the performance seems decent, with inference time around 130ms.

These instructions walk you through building and running the iOS app using Xcode on a Mac. 


### Model
For details about the models we trained and how they were converted to TFLite, please refer to the documentation in the folder src/OCT_Model. There are different Jupyter Notebooks with the steps to train and convert both the ResNet50-V2 and MobileNetV3 models to the TFLite format.



### iOS app details

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift)
to perform the classification of retinal oct images divided in 4 different classes (CNV, DME, DRUSEN and Normal).


## Requirements

*   Device with iOS 12.0 or above

*   Xcode 10.0 or above

*   Valid Apple Developer ID

*   Xcode command-line tools (run `xcode-select --install`)

*   [CocoaPods](https://cocoapods.org/) (run `bash sudo gem install cocoapods`)


If this is a new install, you will need to run the Xcode application once to
agree to the license before continuing.

Note: The demo app requires a camera and must be executed on a real iOS device.
You can build it and run with the iPhone Simulator, but the app will raise a
`Camera not found` exception.

## Build and run

1.  Clone this GitHub repository to your workstation: 
	`bash git clone https://github.com/coyotegil/optical_coherence_tomography.git`

2.  Install the pod to generate the workspace file: 
	`bash cd optical_coherence_tomography/ios && pod install`
	
	- If installing on a Mac with Apple Silicon you'll probably get an error with the command shown above. You can then run the following commands to make the pod install work under rosetta (using intel's binary):
	- `sudo arch -x86_64 gem install ffi`
	- `sudo arch -x86_64 pod install`
		

Note: If you have installed this pod before and that command doesn't work, try
`pod update`.
Again, if using a Mac with Apple Silicon remeber to use: `sudo arch -x86_64 pod update`

At the end of this step you should have a directory called `ImageClassification.xcworkspace`.

1.  Open the project in Xcode with the following command: 
	`bash open ImageClassification.xcworkspace`

This launches Xcode and opens the `ImageClassification` project.

1.  Select the `ImageClassification` project in the left hand navigation to open
    the project configuration. In the **Signing** section of the **General**
    tab, select your development team from the dropdown.

2.  In order to build the project, you must modify the **Bundle Identifier** in
    the **Identity** section so that it is unique across all Xcode projects. To
    create a unique identifier, try adding your initials and a number to the end
    of the string.

3.  With an iOS device connected, build and run the app in Xcode.

You'll have to grant permissions for the app to use the device's camera. Point
the camera at different retinal oct images and see how the model classifies them between CNV, DME, DRUSEN and Normal. You can download some images from the testset of the OCT 2017 dataset we used to train the models to have access to different images pertaining to these classes and to test the app. You can easily find these files on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018). 

The app crops the image from the camera's feed to 224x224 in order to pass it through the model. In order to obtain better results, please try to frame the the oct image at the center of your phone's screen (as shown in the demo video). In our tests we were able to achieve better results by keeping the phone in portrait mode. In the future we intend to add a framing line on the screen to help guide the user through this framing process.

## Model references
The model and classes files are stored in the folder ios/ImageClassification/Model under .tflite and .txt files. The app is setup to use the 'model_oct_resnet50_v2_224.tflite' file but you can easily change it to use the MobileNetV3 model in the code in case you want to give it a try.
