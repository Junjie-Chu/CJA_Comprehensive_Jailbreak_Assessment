#!/bin/bash

pwd
which python

# Define ENV variables
export HF_HOME="PATH_TO_YOUR_HF_CACHE"
export TOKENIZERS_PARALLELISM=true

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <parameter1> <parameter2>"
  echo "Parameter1 can be: llama-guard, llama-guard-2, llama-guard-3, prompt-guard, erase, moderation, perplexity"
  echo "Parameter2 is the input folder path"
  exit 1
fi

# Define the output folder
LOGFOLDER="./logs/"
# Create log folder if it doesn't exist
for folder in $LOGFOLDER; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Folder $folder created."
    else
        echo "Folder $folder already exists."
    fi
done

case "$1" in
    "llama-guard" | "llama-guard-2" | "llama-guard-3")
        chmod +x ./llamaguard/llamaguard_execute.sh
        bash ./llamaguard/llamaguard_execute.sh "${1}" "${2}" > ${LOGFOLDER}/${1}.log 2>&1
        ;;
    "prompt-guard")
        chmod +x ./promptguard/promptguard_execute.sh
        bash ./promptguard/promptguard_execute.sh "${1}" "${2}" > ${LOGFOLDER}/${1}.log 2>&1
        ;;
    "erase")
        chmod +x ./certified-llm-safety/erase_execute.sh
        bash ./certified-llm-safety/erase_execute.sh "${1}" "${2}" > ${LOGFOLDER}/${1}.log 2>&1
        ;;
    "moderation")
        chmod +x ./moderation/moderation_execute.sh
        bash ./moderation/moderation_execute.sh "${1}" "${2}" > ${LOGFOLDER}/${1}.log 2>&1
        ;;
    "perplexity")
        chmod +x ./baseline-defenses/perplexity_filter_execute.sh
        bash ./baseline-defenses/perplexity_filter_execute.sh "${1}" "${2}" > ${LOGFOLDER}/${1}.log 2>&1
        ;;
    *)
        echo "Invalid parameter: $1"
        echo "Valid parameters are: llama-guard, llama-guard-2, llama-guard-3, prompt-guard, erase, moderation, perplexity"
        exit 1
        ;;
esac

echo "Defense ${1} finishes excuting!"