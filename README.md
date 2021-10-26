# Image-Classifier

This is a general image classification model I developed in Tensor Flow.
Currently you are able to train a general model and evaluate this model. I am also working on a way for the user to actually use the model to predict data.

# Setup
To run these programs you'll need to install ensure you have python installed here's a link for installing python (https://www.python.org/downloads/).
Once you have python installed youll also need to install Tensoer Flow, at least version 2.6.0 you can do this by running "pip install tensorflow" within the project directory. Once you have done this you are good to go!

#### Note If you wish to use cuda (Nvidia Graphics Cards) youll need to follow this guide (https://www.tensorflow.org/install/gpu) though ensure you follow the guide exactly, ensure you get the correct versions of each piece of software other wise it will not work. In my case I was unable to add cuda to my path, if this is the case for you, add these lines of code below to the top of your python files.
##### import os
##### os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin") (Or use the path to where you installed cuda)

# Running Train
To run training of the model you can simply input all of your data into the "trainData" directory following a directory format specified below and run "python ./train.py".

### trainData
>#### Class 1
>>##### image 1
>> ##### image ...
>>
>#### Class 2
>>##### image 1
>>##### image ...

This will train a model for you and output it to model.pth, but your accuracy may vary, have fun messing around with different parameters within the model and try to get the best accuracy for your data.

# Running Evaluate
Running Evaluate on your model will allow you to see the Loss, Accuracy, Recall, Precision and F1 score. You can find what each of these metrics mean online.
To run evaluate simply input all your data into "dataToEvaluate" following a directory structure shown below:

### dataToEvaluate
>#### Class 1
>>##### image 1
>> ##### image ...
>>
>#### Class 2
>>##### image 1
>>##### image ...

Once you've run this the program will print to consile the relevant information.

# Running Predict
## This feature is currently unavailable.
