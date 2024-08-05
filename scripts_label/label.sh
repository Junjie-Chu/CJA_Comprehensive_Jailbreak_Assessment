#!/bin/bash

cd PATH/scripts_label
which python

# The directory containing the JSON files (jailbreak responses) to be labeled
DIRECTORY=$1

# Count the number of JSON files in the directory
TOTAL_FILES=$(find "$DIRECTORY" -maxdepth 1 -name '*.json' | wc -l)
echo "In total $TOTAL_FILES JSON files."

# Current file counter
COUNTER=0

# The path to the forbidden questions CSV file
QUESTIONS='../forbidden_questions/forbidden_questions.csv'

# Iterate over all the files in the directory
for file in "$DIRECTORY"/*.json
do
    # Update the counter
    ((COUNTER++))
    # Check if the file exists
    if [ -f "$file" ];then
        echo "Now processing: $COUNTER / $TOTAL_FILES: $file"
        # pass the result file name to the py program
        # start_line range [0,159]
        python label.py \
        --model_name gpt-4-turbo \
        --test_mode False \
        --start_line 0 \
        --raw_questions_path "$QUESTIONS" \
        --results_path "$file"
    fi
done

echo "All files have been labeled."

