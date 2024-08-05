# CJA_Comprehensive_Jailbreak_Assessment
This is the public code repository of paper 'Comprehensive Assessment of Jailbreak Attacks Against LLMs.'

## How to use this repository?
### Install and set the ENV
1. Clone this repository.
2. ```conda create -n CJA python=3.9```
3. ```pip install -r requirements.txt```
### Use our labeling method to label your own results
1. Switch directory:```cd ./scripts_label```
2. ```python label.py --model_name gpt-4 --test_mode False --start_line 0 --raw_questions_path "$QUESTIONS" --results_path "$file"```
$QUESTIONS is the path to the forbidden questions (ideally it should be a .csv file, refer to ./forbidden_questions/forbidden_questions.csv for example).
$file is the path to the LLM responses after jailbreak, it should be a .json file.
The .json file could be generated by the following codes, note that *answer* is the response from the target LLM. 
```
answers.append({'response': answer})

# Write into the output file
with open(output_file, 'w') as out_file:
    json.dump(answers, out_file, indent=4)
```
**Optional**
You may also utilize label.sh to label files in a directory.
```bash label.sh PATH_TO_RESPONSES_DIRECTORY```
The files storing the labels will be save to the same directory where you store the jailbreak responses.