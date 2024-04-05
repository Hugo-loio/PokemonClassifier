import sys
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)  # Decode the image with 3 channels (RGB)
    img = tf.image.resize(img, [*image_size])  # Resize the image
    img = tf.expand_dims(img, axis=0) # Add batch dimension
    return img

# Define directories
rootdir = sys.path[0]
datadir = rootdir + "/data"
plot_dir = rootdir + "/plots"
model_dir = rootdir + "/models"

# Get model
model_name = 'model2'
model = load_model(model_dir + '/' + model_name + '.h5')
#model.summary()

image_size = model.layers[0].output_shape[1:3]

if(len(sys.argv) == 1):
    print("Please input a path for an image")
    exit()

image_path = sys.argv[1]
image = preprocess_image(image_path)

#plt.imshow(image[0].numpy().astype("uint8"))
#plt.show()

prediction = model.predict(image)
class_names = sorted(os.listdir(datadir + "/training"))

predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index]

print("Predicted Pokemon:", predicted_class)
print("Confidence:", confidence)
