import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin") #allows me to run on cuda remove if this is giving error
import tensorflow as tf

model = tf.keras.models.load_model('model.pth') # loads the model

testDataDir = "./dataToEvaluate" #evaluating data directory

image_size_x=300
image_size_y=300

test_dir = tf.keras.utils.image_dataset_from_directory( #loads the test data
    testDataDir,label_mode="categorical",image_size=(image_size_x,
    image_size_y),batch_size=8,seed=309
)

results = model.evaluate(test_dir) #evaluates the model based on the test data

#prints all relavent information
print("Loss:"+str(results[0]))
print("accuracy:"+str(results[1]))
print("recall:"+str(results[2]))
print("precision:"+str(results[3]))

f1score = 2 * ((results[3] * results[2])/(results[3] + results[2])) #calculates the f1 score

print("F1-Score:"+str(f1score))