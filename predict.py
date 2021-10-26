import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin") #allows me to run on cuda remove if this is giving error
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join

model = tf.keras.models.load_model('model.pth',compile = True) # loads the model

testDataDir = "./dataToPredict" #testing data directory

f = open("labels.txt", "r")

labels = []

line = f.readline().split(",")

print(len(line))

for val in line:
    labels.append(val);

print(labels)
image_size_x=300
image_size_y=300

onlyfiles = [f for f in listdir(testDataDir) if isfile(join(testDataDir, f))]
print(onlyfiles)

for f in onlyfiles:
    image = tf.keras.preprocessing.image.load_img(testDataDir+"/"+f)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    print(predictions)
    preds = np.argmax(predictions, axis=1)
    print(preds)
    print(labels[preds[0]])





