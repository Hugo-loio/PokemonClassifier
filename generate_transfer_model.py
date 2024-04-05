import sys
import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
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
image_size = (100,100)
epochs = 15 
padding = 'valid'
model_name = "final_transfer_model"

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

resize = tf.keras.Sequential([layers.Resizing(*image_size)])

rescale = tf.keras.Sequential([layers.Rescaling(1./255)])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.2)
    ])

input_shape =   image_size + (3,)

# Load pretrained convolutional base (e.g., VGG16)
conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the convolutional base
conv_base.trainable = False

model = models.Sequential([
    conv_base,

    layers.Flatten(),
    layers.Dropout(0.5), 

    layers.Dense(1024, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.5),  

    layers.Dense(512, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.5), 

    layers.Dense(num_classes, activation='softmax')
    ])

model.build(input_shape)
model.summary()

model.compile(
    optimizer=optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

utils.plot_model(model, show_shapes=True, to_file=plot_dir + '/' + model_name + '_scheme.png')

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
