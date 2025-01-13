#!/bin/bash

# Get a directory from the command line
directory=$1

# Loop through the list of files in the directory
for file in "$directory"/*; do
  echo "Processing file: $file"
  # Run the Python script with the current file as an argument
  python src/llm.py "$file"
done