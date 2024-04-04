import sys
import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models
print("Keras version:", tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#import numpy as np
import matplotlib.pyplot as plt

def get_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
            directory, 
            labels = 'inferred', 
            #label_mode = 'categorical',
            shuffle = True,
            image_size = image_size,
            batch_size = batch_size
            )


# Define directories
rootdir = sys.path[0]
datadir = rootdir + "/data"
training_dir = datadir + "/training"
validation_dir = datadir + "/validation"
test_dir = datadir + "/test"
plot_dir = rootdir + "/plots"
model_dir = rootdir + "/models"
for directory in [plot_dir, model_dir]:
    if(not os.path.isdir(directory)):
        os.mkdir(directory)

# Set some hyperparameters
batch_size = 32
image_size = (256, 256)
epochs = 10 
model_name = "model1"

# Get datasets and classes
training_dataset = get_dataset(training_dir)
pokemons = training_dataset.class_names
num_classes = len(pokemons)

validation_dataset = get_dataset(validation_dir)
test_dataset = get_dataset(test_dir)

# Output a couple of random images from the training dataset
fig = plt.figure()
for image_batch, label_batch in training_dataset.take(1):
    for i in range(0,9):
        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(image_batch[i].numpy().astype("uint8"))
        ax.set_title(pokemons[label_batch[i]])
        ax.axis("off")
fig.savefig(plot_dir + "/image_check.png", bbox_inches = "tight", dpi = 300)


# Configure the datasets for performance
training_dataset = training_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

'''
resize_and_rescale = tf.keras.Sequential([
    #layers.experimental.preprocessing.Resizing(image_size),
    layers.experimental.preprocessing.Rescaling(1./255),
    ])
    '''

rescale = tf.keras.Sequential([
    layers.Rescaling(1./255)
    ])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomContrast(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2)
    ])

input_shape = (batch_size, ) + image_size + (3,)

model = models.Sequential([
    rescale,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), 
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])

model.build(input_shape)
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
        training_dataset, 
        batch_size =  batch_size,
        epochs=epochs, 
        validation_data=validation_dataset
        )

# Plot accuracies
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='training')
ax.plot(history.history['val_accuracy'], label = 'validation')
ax.set(xlabel = 'Epoch', ylabel = 'Accuracy')
ax.legend()
fig.savefig(plot_dir + "/" + model_name + "_accuracy.pdf", bbox_inches = "tight")

# Plot losses
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label = 'training')
ax.plot(history.history['val_loss'], label = 'validation')
ax.set(xlabel = 'Epoch', ylabel = 'Loss')
ax.legend()
fig.savefig(plot_dir + "/" + model_name + "_loss.pdf", bbox_inches = "tight")

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

model.save(model_dir + "/" + model_name + ".h5")
with open(model_dir + "/" + model_name + "_history.pickle", 'wb') as pickle_file:
    pickle.dump(history, pickle_file)
