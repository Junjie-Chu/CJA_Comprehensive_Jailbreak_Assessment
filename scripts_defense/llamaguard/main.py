import os
# Set the cache directory path
cache_dir_path = "PATH_TO_YOUR_HF_CACHE"
os.environ["HF_HOME"] = cache_dir_path
import argparse
import json
import csv
import pandas as pd
from vllm import LLM, SamplingParams
import openai
from tqdm import tqdm  # Import tqdm
import backoff 
import pandas as pd
from transformers import AutoTokenizer,LlamaTokenizer
import tiktoken

def message_generation_llamaguard(adv_prompt):
    message = [ {
                "role": "user",
                "content": adv_prompt
                },
                ]
    return message

def main(guard, input_file_path, output_folder_path):

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

    # Create an LLM.
    model_dict = {
                "llama-guard":"PATH_TO_YOUR_HF_CACHE/meta-llama/LlamaGuard-7b",
                "llama-guard-2":"PATH_TO_YOUR_HF_CACHE/meta-llama/Meta-Llama-Guard-2-8B",
                "llama-guard-3":"PATH_TO_YOUR_HF_CACHE/meta-llama/Llama-Guard-3-8B",
                }

    model_name = guard
    llm = LLM(model=model_dict[model_name],trust_remote_code=False,tokenizer_mode="auto")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dict[model_name])

    # llamaguard dict
    # llamaguard_dict = {"O1":"Violence and Hate","O2":"Sexual Content","O3":"Criminal Planning","O4":"Guns and Illegal Weapons","O5":"Regulated or Controlled Substances","O6":"Self-Harm"}

    with open(input_file_path, 'r') as f:
        data = json.load(f)

    sequences = data if isinstance(data, list) else data['sequences']

    harms = []
    for i in range(len(sequences)):
        prompts = []
        sequence = sequences[i]
        chat = message_generation_llamaguard(adv_prompt = sequence)
        tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(tokenized_chat)

        print(f'Batch Size = {len(prompts)} \n')
        outputs = llm.generate(prompts, sampling_params)

        for j in range(len(prompts)):
            # Update temp_results with new data
            prompt = outputs[j].prompt
            generated_text = outputs[j].outputs[0].text.strip()
            if generated_text == 'safe':
                # Not ADV = 0
                print(f'Not ADV = 0, output: {generated_text}')
                attack_detected = False
            else:
                # ADV = 1
                print(f'ADV = 1, output: {generated_text}')
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
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(['prompt', 'defense_status'])
        
        for sequence, harm in zip(sequences, harms):
            csvwriter.writerow([sequence, harm])

    print(f"Results have been saved to {output_csv_path} ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter sequences based on safetyguard.")
    parser.add_argument('--guard', type=str, help="Guard model to use.")
    parser.add_argument('--input_file_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_folder_path', type=str, help="Path to the output folder.")
    
    args = parser.parse_args()
    
    main(args.guard, args.input_file_path, args.output_folder_path)