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

# Get test dataset
image_size = (256, 256)
test_dataset = get_dataset(test_dir)

# Get model
model_name = 'model1'
model = load_model(model_dir + '/' + model_name + '.h5')

with open(model_dir + "/" + model_name + "_history.pickle", 'rb') as pickle_file:
    history = pickle.load(pickle_file)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='training')
ax.plot(history.history['val_accuracy'], label = 'validation')
ax.set(xlabel = 'Epoch', ylabel = 'Accuracy')
ax.legend()
fig.savefig(plot_dir + "/" + model_name + "_accuracy.pdf", bbox_inches = "tight")

fig, ax = plt.subplots()
ax.plot(history.history['loss'], label = 'training')
ax.plot(history.history['val_loss'], label = 'validation')
ax.set(xlabel = 'Epoch', ylabel = 'Loss')
ax.legend()
fig.savefig(plot_dir + "/" + model_name + "_loss.pdf", bbox_inches = "tight")

#print("Test loss: ", test_loss, "Test accuracy: ", test_acc)

