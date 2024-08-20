import os
# Set the cache directory path
cache_dir_path = "PATH_TO_YOUR_HF_CACHE"
os.environ["HF_HOME"] = cache_dir_path
import argparse
import torch
import json
import csv
import pandas as pd
import openai
from tqdm import tqdm  # Import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tiktoken

def main(guard, input_file_path, output_folder_path):

    # Create an LLM.
    model_dict = {
                "prompt-guard":"PATH_TO_YOUR_HF_CACHE/meta-llama/Prompt-Guard-86M",
                }

    model_name = guard
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dict[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(model_dict[model_name])

    with open(input_file_path, 'r') as f:
        data = json.load(f)

    sequences = data if isinstance(data, list) else data['sequences']

    harms = []
    for i in range(len(sequences)):
        prompts = []
        sequence = sequences[i]
        text = sequence
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        predicted_class_id = logits.argmax().item()
        label = model.config.id2label[predicted_class_id]

        if label == 'BENIGN' or label == 'INJECTION':
            # Not-ADV = 0, BENIGN or INJECTION
            print(f'Not ADV = 0, output: {label}')
            attack_detected = False
        else:
            # ADV = 1, JAILBREAK
            print(f'ADV = 1, output: {label}')
            attack_detected = True
        harms.append(attack_detected)

    # Prepare output path
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)
    
    # Form the output filename based on input filename and path
    input_basename = os.path.basename(input_file_path)
    output_basename = input_basename.replace('.json', f'_defense_{model_name}.csv')
    output_csv_path = os.path.join(output_folder_path, output_basename)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # CSV object
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(['prompt', 'defense_status'])
        
        # Iterate
        for sequence, harm in zip(sequences, harms):
            csvwriter.writerow([sequence, harm])

    print(f"Results have been saved to {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter sequences based on safetyguard.")
    parser.add_argument('--guard', type=str, help="Guard model to use.")
    parser.add_argument('--input_file_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_folder_path', type=str, help="Path to the output folder.")
    
    args = parser.parse_args()
    
    main(args.guard, args.input_file_path, args.output_folder_path)