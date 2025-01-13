#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -d <directory> -p <num_partitions>"
    echo "Options:"
    echo "  -d <directory>       Directory to partition"
    echo "  -p <num_partitions>  Number of partitions to create"
    exit 1
}

# Variables
DIRECTORY=""
NUM_PARTITIONS=0

# Parse command-line arguments
while getopts ":d:p:" opt; do
    case $opt in
        d) DIRECTORY=$OPTARG ;;
        p) NUM_PARTITIONS=$OPTARG ;;
        *) usage ;;
    esac
done

# Validate inputs
if [[ -z "$DIRECTORY" || $NUM_PARTITIONS -le 0 ]]; then
    usage
fi

if [[ ! -d "$DIRECTORY" ]]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Create base partition directory in /tmp
BASE_PARTITION_DIR="/tmp/partition_$(basename $DIRECTORY)"
mkdir -p "$BASE_PARTITION_DIR"

# Get a list of all files in the directory (skip directories)
FILES=()
while IFS= read -r -d $'\0'; do
    FILES+=("$REPLY")
done < <(find "$DIRECTORY" -type f -print0)

TOTAL_FILES=${#FILES[@]}
if [[ $TOTAL_FILES -eq 0 ]]; then
    echo "Error: No files found in $DIRECTORY."
    exit 1
fi

# Calculate files per partition
FILES_PER_PARTITION=$((TOTAL_FILES / NUM_PARTITIONS))
REMAINING_FILES=$((TOTAL_FILES % NUM_PARTITIONS))

echo "Partitioning $TOTAL_FILES files into $NUM_PARTITIONS partitions..."

# Distribute files across partitions
INDEX=0
for PARTITION_NUMBER in $(seq 1 $NUM_PARTITIONS); do
    PARTITION_DIR="$BASE_PARTITION_DIR/partition_$PARTITION_NUMBER"
    mkdir -p "$PARTITION_DIR"

    COUNT=$FILES_PER_PARTITION
    # Distribute the remaining files one by one to partitions
    if [[ $REMAINING_FILES -gt 0 ]]; then
        COUNT=$((COUNT + 1))
        REMAINING_FILES=$((REMAINING_FILES - 1))
    fi

    # Copy files to the current partition
    for ((i = 0; i < COUNT; i++)); do
        if [[ $INDEX -lt $TOTAL_FILES ]]; then
            cp "${FILES[$INDEX]}" "$PARTITION_DIR/"
            INDEX=$((INDEX + 1))
        fi
    done
done

echo "Partitioning completed. Partitions created in $BASE_PARTITION_DIR."