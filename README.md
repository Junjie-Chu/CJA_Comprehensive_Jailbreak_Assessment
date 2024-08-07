# Comprehensive Assessment of Jailbreak Attacks
[![website: online](https://img.shields.io/badge/website-online-blue.svg)](https://junjie-chu.github.io/Public_Comprehensive_Assessment_Jailbreak/)
[![dataet: released](https://img.shields.io/badge/dataset-released-green.svg)](https://github.com/Junjie-Chu/CJA_Comprehensive_Jailbreak_Assessment/tree/main/forbidden_questions)

This is the public repository of paper [*Comprehensive Assessment of Jailbreak Attacks Against LLMs*](https://arxiv.org/abs/2402.05668). 

## How to use this repository?
### Install and set the ENV
1. Clone this repository.
2. Prepare the python ENV.
```
conda create -n CJA python=3.9
conda activate CJA
cd PATH_TO_THE_REPOSITORY
pip install -r requirements.txt
```
### Use our labeling method to label your own results

**Option 1: label single file**  
1. Switch directory:  
```
cd ./scripts_label
```
2. Command to label single file:
```
python label.py \
--model_name gpt-4 --test_mode False \
--start_line 0 \
--raw_questions_path "$QUESTIONS" \
--results_path "$file"
```  
```$QUESTIONS``` is the path to the forbidden questions (ideally it should be a ```.csv``` file, refer to ./forbidden_questions/forbidden_questions.csv for example).  
```$file``` is the path to the LLM responses after jailbreak, it should be a ```.json``` file. The ```.json``` file could be generated by the following codes.
```
answers.append({'response': answer})
# Write into the output file
with open(output_file, 'w') as out_file:
    json.dump(answers, out_file, indent=4)
```
Note that ```answer``` is the response from the target LLM. 

**Option 2: label files in a directory**  
You may also utilize label.sh to label files in a directory:  
```
bash label.sh PATH_TO_RESPONSES_DIRECTORY
```
***The files storing the labels will be save to the same directory where you store the jailbreak responses.*** 

## Add new results to the leaderboard.
Welcome to submit your own evaluation results (steps = 50) to us. 
The leaderboard is available [here](https://junjie-chu.github.io/Public_Comprehensive_Assessment_Jailbreak/leaderboard).

*Full codes will be released after the papaer is accepted.* 
