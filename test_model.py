import sys
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

#import numpy as np
import matplotlib.pyplot as plt

def get_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
            directory, 
            labels = 'inferred', 
            #label_mode = 'categorical',
            shuffle = True,
            image_size = image_size
            )


# Define directories
rootdir = sys.path[0]
datadir = rootdir + "/data"
test_dir = datadir + "/test"
plot_dir = rootdir + "/plots"
model_dir = rootdir + "/models"

# Get model
model_name = 'final_cnn_model'
print("Testing model:", model_name)
model = load_model(model_dir + '/' + model_name + '.h5')
model.summary()

# Get test dataset
image_size = model.layers[0].output_shape[1:3]
test_dataset = get_dataset(test_dir)

# Evaluate test dataset
print("Test evaluation:")
model.evaluate(test_dataset, verbose=2)

#with open(model_dir + "/" + model_name + "_history.pickle", 'rb') as pickle_file:
#    history = pickle.load(pickle_file)
