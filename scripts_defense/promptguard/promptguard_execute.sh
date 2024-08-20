#!/bin/bash

which python
pwd

# Switch work space folder
cd ./promptguard

# Define the output folder
OUTPUTFOLDER="../../../CJA_results/jailbreak_defense/${1}/"

# Create results folder if it doesn't exist
for folder in $OUTPUTFOLDER; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Folder $(realpath "$folder") created."
    else
        echo "Folder $(realpath "$folder") already exists."
    fi
done

INPUTFOLDER=${2}

TOTAL_FILES=$(find "$INPUTFOLDER" -maxdepth 1 -name '*.json' | wc -l)

echo "There are $TOTAL_FILES JSON files found in ${INPUTFOLDER}."

COUNTER=0
# Define the file paths
for FILE_PATH in "$INPUTFOLDER"/*.json
do
    ((COUNTER++))

    if [ -f "$FILE_PATH" ]
    then
        echo "Processing $COUNTER / $TOTAL_FILES: $FILE_PATH"
        python -u main.py \
        --guard "${1}" \
        --input_file_path "$FILE_PATH" \
        --output_folder_path "$OUTPUTFOLDER"
    fi
done

echo "All files have been saved!"