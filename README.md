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
1.31 Gb of data will be downloaded from [Ben Hawks'](https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k/data) data set.
Additionally, a small data set of 4Mb from user [Vishal Subbiah](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/data) is downloaded to provide a clean image for each Pokémon, for presentation purposes.

## Model generation

To train the model, run `python generate_model.py`.
This can take some time if a GPU is not used so pre-trained models are already included in the `models/` directory.
Once the model is generated, the history plots of fitting will be saved in the generated `plots/` directory.


A model based on transfer learning from the VGG16 model can also be generated with `python generate_transfer_model.py`.

The model generation takes a couple of minutes to run in an NVDIA RTX A1000.
## Testing

Saved models can be tested against the testing dataset using the `test_model.py` script.
To use the models for prediction run `python predict_pokemon.py <path_to_image>`.
The script `predict_random_pokemon.sh` loops through random Pokémon in the testing dataset and uses the model indicated in `predict_pokemon.py`.

In these python scripts, the variable `model_name` might need to be changed to use different models. 
By default, it is set to `'final_cnn_model'`, corresponding to the default name of the convolution neural network model.
See which models are available in the `models` directory.
