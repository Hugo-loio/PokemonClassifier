This is a project for the course "Machine (Deep) Learning for physicists 2023-24 " taught by Prof. Rudo Roemer at university of Warwick.
It is a deep learning model for classifying generation one Pokémon from images.

# Dependencies

A working local environment with the following libraries:
* Tensorflow for python.
* Standard python libraries: NumPy and Matplotlib
* The [Kaggle API](https://www.kaggle.com/docs/api) for downloading the data sets

# User guide

## Data set generation

To download the data set and split it into training, validation and testing data sets run `python generate_dataset.py`. 
A local installation of the Kaggle API is necessary.
1.31 Gb of data will be download from [Ben Hawks'](https://www.kaggle.com/data sets/bhawks/pokemon-generation-one-22k/data) data set.
Additionally, a small data set of 4Mb from user [Vishal Subbiah](vishalsubbiah/pokemon-images-and-types) is downloaded to provide a clean image for each Pokémon, for presentation purposes.

## Model generation

To train the model, run `python generate_model.py`.
This can take some time if a GPU is not used so pre-trained models are already included in the `models/` directory.
