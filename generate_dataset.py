import os 
import sys
import shutil
import random

def delete_files_with_wrong_extension():
    # Get list of files in the folder
    pokemons = os.listdir(dataset_dir)
    for pokemon in pokemons:
        pokemon_dir = dataset_dir + "/" + pokemon
        images = os.listdir(pokemon_dir)

        for image in images:
            _, file_extension = os.path.splitext(image)
            if file_extension.lower() not in ('.jpg', '.jpeg', '.png', '.svg'):
                image_path = pokemon_dir + "/" + image
                os.remove(image_path)
                print(f"Deleted: {image_path}")

def split_data():
    # Get list of files in the folder
    pokemons = os.listdir(dataset_dir)
    count_training = 0
    count_validation = 0
    count_total = 0
    for pokemon in pokemons:
        training_pokemon_dir = training_dir + "/" + pokemon
        os.mkdir(training_pokemon_dir)
        
        validation_pokemon_dir = validation_dir + "/" + pokemon
        os.mkdir(validation_pokemon_dir)

        test_pokemon_dir = test_dir + "/" + pokemon
        os.mkdir(test_pokemon_dir)

        pokemon_dir = dataset_dir + "/" + pokemon
        images = os.listdir(pokemon_dir)
        random.shuffle(images)

        total = len(images)
        count_total += total
        training_index = int(training*total)
        count_training += training_index
        validation_index = int((training + validation)*total)
        count_validation += validation_index - training_index

        for image in images[:training_index]:
            image_path = pokemon_dir + "/" + image
            shutil.copy(image_path, training_pokemon_dir)

        for image in images[training_index:validation_index]:
            image_path = pokemon_dir + "/" + image
            shutil.copy(image_path, validation_pokemon_dir)

        for image in images[validation_index:]:
            image_path = pokemon_dir + "/" + image
            shutil.copy(image_path, test_pokemon_dir)

    print("Dataset spliting")
    print("Training:", 100*count_training/count_total, "%")
    print("Validating:", 100*count_validation/count_total, "%")
    print("Testing:", 100 - 100*(count_validation + count_training)/count_total, "%")
    

# Call the function

rootdir = sys.path[0]
datadir = rootdir + "/data"
dataset_dir = datadir + "/PokemonData"

if(os.path.isdir(datadir)):
    shutil.rmtree(datadir)

os.system("kaggle datasets download -d lantian773030/pokemonclassification")

os.mkdir(datadir)

# Unzip archive
shutil.unpack_archive(rootdir + "/pokemonclassification.zip", datadir)

# Cleanup data
shutil.rmtree(dataset_dir + "/Alolan Sandslash")
delete_files_with_wrong_extension()

training_dir = datadir + "/training"
os.mkdir(training_dir)
validation_dir = datadir + "/validation"
os.mkdir(validation_dir)
test_dir = datadir + "/test"
os.mkdir(test_dir)

training = 0.81 # 80 % of the data
validation = 0.15 # 15 % of the data
test = 0.04 # 5 % of the data

split_data()
