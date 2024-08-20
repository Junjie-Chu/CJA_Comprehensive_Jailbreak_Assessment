import os
# Set the cache directory path
cache_dir_path = "PATH_TO_YOUR_HF_CACHE"
os.environ["HF_HOME"] = cache_dir_path
import argparse
import json
import csv
import pandas as pd
from tqdm import tqdm  # Import tqdm
import backoff 
import pandas as pd
import tiktoken
from openai import OpenAI

client = OpenAI(api_key = "YOUR_OPENAI_KEY_HERE")

def openai_moderation_measure(text:str):
    response = client.moderations.create(input=text)
    response_dict = response.model_dump()
    flag = response_dict['results'][0]['flagged']
    return flag

def main(input_file_path, output_folder_path):
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    sequences = data if isinstance(data, list) else data['sequences']

    harms = []
    for i in range(len(sequences)):
        sequence = sequences[i]
        flag = openai_moderation_measure(sequence)

        if flag == False:
            # Not ADV = 0
            print(f'Not ADV = 0, output: {flag}')
            attack_detected = False
        else:
            # ADV = 1
            print(f'ADV = 1, output: {flag}')
            attack_detected = True
        harms.append(flag)

    # Prepare output path
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)
    
    # Form the output filename based on input filename and path
    input_basename = os.path.basename(input_file_path)
    output_basename = input_basename.replace('.json', '_defense_moderation.csv')
    output_csv_path = os.path.join(output_folder_path, output_basename)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(['prompt', 'defense_status'])
        
        for sequence, harm in zip(sequences, harms):
            csvwriter.writerow([sequence, harm])

    print(f"Results have been saved to {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter sequences based on moderation.")
    parser.add_argument('--input_file_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_folder_path', type=str, help="Path to the output folder.")
    
    args = parser.parse_args()
    
    main(args.input_file_path, args.output_folder_path)