import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
from tensorflow.keras import  layers, models

trainDataDir = "./trainData" #training data directory

testDataDir = "./trainData" #testing data directory

image_size_x=300
image_size_y=300
seed = 309
batchSize = 8




train_dir = tf.keras.utils.image_dataset_from_directory( #loads the training data sets This will take a train split from 
    trainDataDir,label_mode="categorical",image_size=(image_size_x,
    image_size_x),batch_size=batchSize,seed=seed,validation_split=0.2,subset="training")

test_dir = tf.keras.utils.image_dataset_from_directory( #loads the test data sets
    testDataDir,label_mode="categorical",image_size=(image_size_x,
    image_size_x),batch_size=batchSize,seed=seed,validation_split=0.2,subset="validation"
)


model = models.Sequential() #creates a new sequential model

model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(image_size_x, image_size_x, 3))) 
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((8, 8))) 
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((16, 16))) 
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation="relu"))  
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(len(train_dir.class_names), activation='softmax')) 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                            loss=tf.keras.losses.CategoricalCrossentropy(),
                            metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.summary() # prints summary of model

history = model.fit(train_dir, epochs=15, #trains and fits the model
                            validation_data=(test_dir))

f = open("labels.txt","w") #Writes labels to a file
for val in train_dir.class_names:
    f.write(val+",")
f.close()

model.save("model.pth"); #saves the model

