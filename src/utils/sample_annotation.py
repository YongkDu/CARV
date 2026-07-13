"""
Sample wrong-answer cases from inference outputs for human annotation.

Reads output JSONL files, filters to items where the model answered incorrectly,
and saves a random sample to error_ana/data/{task}_hard/ for the human
annotation study.
"""
import os
from utils import read_jsonl
import json

# modle_list = ['gemini2.5pro', 'gemini2.5flash', 'gpt', 'gpt5']
modle_list = ['qwen3-30b']
task = 'mix'
output_root = '/data/yongka/analogy/spatial/output_base_new/'

# num_per_model = {'gpt': 75, 'gpt5': 76, 'gemini2.5pro': 90, 'gemini2.5flash': 76}  # 'gemini2.5pro': 50, 'gemini2.5flash': 50}
num_per_model = {'gpt': 75, 'gpt5': 75, 'gemini2.5pro': 75, 'gemini2.5flash': 75}  # 'gemini2.5pro': 50, 'gemini2.5flash': 50}
num_per_model = {'qwen3-30b': 75}  # 'gemini2.5pro': 50, 'gemini2.5flash': 50}

wrong_data = ['two_toilet-roll_left_of_blue_chair', 'two_toilet-roll_left_of_red_chair', 'two_toilet-roll_left_of_wood_chair',]

for task in ['mix']:
    for operation in ['union', 'intersection']:
        for model_name in modle_list:
            sampled_data = []
            file = f"{output_root}/{task}/{operation}/{model_name}_easy_500.jsonl"
            data = read_jsonl(file)
            count = 0
            skip_count = 0
            for item in data:
                skip = False
                for wrong in wrong_data:
                    if wrong in item['context_images']:
                        skip = True
                        skip_count += 1
                        break
                if skip:
                    continue
                if item['correctness_analysis']['correctness'] == False:
                    sampled_data.append(item)
                    count += 1
                if count >= num_per_model[model_name]:
                    break
            # if len(sampled_data) < num_per_model[model_name]:
            #     file = f"{output_root}/{task}/{operation}/{model_name}_hard_500.jsonl"
            #     data = read_jsonl(file)
            #     for item in data:
            #         skip = False
            #         for wrong in wrong_data:
            #             if wrong in item['context_images']:
            #                 skip = True
            #                 break
            #         if skip:
            #             continue
            #         if item['correctness_analysis']['correctness'] == False:
            #             sampled_data.append(item)
            #             count += 1
            #         if count >= num_per_model[model_name]:
            #             break
            print(f"Task: {task}, Sampled error cases: {len(sampled_data)}, Model: {model_name}, Operation: {operation}, Skipped: {skip_count}")
            save_folder = f"/data/yongka/analogy/spatial/error_ana/data/{task}_hard"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = f"{save_folder}/{operation}_{model_name}.jsonl"
            with open(save_path, 'w') as f:
                for item in sampled_data:
                    f.write(json.dumps(item) + '\n')