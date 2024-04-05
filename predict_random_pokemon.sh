#!/bin/sh

pokemon=$(ls data/test)
nr_pokemon=$(ls data/test | wc -l)

while true; do
  random_number=$(shuf -i1-$nr_pokemon -n1)
  random_pokemon=$(echo $pokemon | cut -d " " -f $random_number)

  images=$(ls data/test/$random_pokemon)
  nr_images=$(ls data/test/$random_pokemon | wc -l)

  random_number=$(shuf -i1-$nr_images -n1)
  random_image=$(echo $images | cut -d " " -f $random_number)

  echo "\nTesting Pokémon: $random_pokemon"
  python predict_pokemon.py data/test/$random_pokemon/$random_image

  #read -p "Would you like to continue testing? (y/n)" yn
  #case $yn in
  #    [Yy]* ) continue;;
  #esac
  #exit
done
