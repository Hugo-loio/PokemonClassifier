#!/bin/sh

pokemons=$(ls data/test)

for pokemon in $pokemons; do
  echo "\nTesting $pokemon..."
  images=$(ls data/test/$pokemon)
  for image in $images; do
    python guess_pokemon.py data/test/$pokemon/$image
  done
done
