#!/bin/bash

# Get a list of files from the command line arguments
files=("$@")

# Loop through the list of files
for file in "${files[@]}"; do
  echo "Processing file: $file"
  # Run the Python script with the current file as an argument
  python src/llm.py "$file"
done