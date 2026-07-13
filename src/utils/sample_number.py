"""
Sample error cases filtered by property combination for property-level analysis.

Reads output JSONL files and filters to incorrect predictions grouped by
property combination (e.g. subject_position). Saves results under
error_ana/property/ for use by analysis_3step_property.py.
"""
import os
from utils import read_jsonl
import json

modle_list = ['gemini2.5flash'] #  
task = 'mix'
output_root = '/data/yongka/analogy/spatial/output_base_new'
label_root = f'/data/yongka/analogy/spatial/data_labels_all'
property_dict = {
                    'union': {'gpt': ['number', 'color'], 'gpt5': ['number', 'subject'], 'gemini2.5pro': ['number', 'position'], 'gemini2.5flash': ['object', 'color']},
                    'intersection': {'gpt': ['number', 'subject'], 'gpt5': ['number', 'position'], 'gemini2.5pro': ['number', 'position'], 'gemini2.5flash': ['object', 'color']}
                }  # 'gemini2.5pro': 50, 'gemini2.5flash': 50}

wrong_data = ['two_toilet-roll_left_of_blue_chair', 'two_toilet-roll_left_of_red_chair', 'two_toilet-roll_left_of_wood_chair',]
# data_size = 50

properties = [
    ['number', 'position'],
    ['number', 'subject'],
    ['subject', 'position'],
    ['object', 'color']]
for p in properties:
    sampled_data = []
    for task in ['mix']:
        for model_name in modle_list:
            for operation in ['intersection', 'union']:
                file = f"{output_root}/{task}/{operation}/{model_name}_easy_500.jsonl"
                data = read_jsonl(file)
                label = json.load(open(f"{label_root}/{task}/{operation}/easy_500.json"))
                count = 0
                skip_count = 0
                for i, item in enumerate(data):
                    skip = False
                    for wrong in wrong_data:
                        if wrong in item['context_images']:
                            skip = True
                            skip_count += 1
                            break
                    if skip:
                        continue
                    if item['correctness_analysis']['correctness'] == False and (sorted(label[i]['changed_properties1']) == sorted(p) or sorted(label[i]['changed_properties2']) == sorted(p)):
                        sampled_data.append(item)
                        count += 1
                    # if count >= data_size:
                    #     break
                    # if len(sampled_data) < data_size:
                    file = f"{output_root}/{task}/{operation}/{model_name}_hard_500.jsonl"
                    data = read_jsonl(file)
                    label = json.load(open(f"{label_root}/{task}/{operation}/hard_500.json"))
                    for i, item in enumerate(data):
                        skip = False
                        for wrong in wrong_data:
                            if wrong in item['context_images']:
                                skip = True
                                skip_count += 1
                                break
                        if skip:
                            continue
                        if item['correctness_analysis']['correctness'] == False and (sorted(label[i]['changed_properties1']) == sorted(p) or sorted(label[i]['changed_properties2']) == sorted(p)):
                            sampled_data.append(item)
                            count += 1
                        # if count >= data_size:
                        #     break
            print(f"Task: {task}, Sampled error cases: {len(sampled_data)}, Model: {model_name}, Operation: {operation}, Skipped: {skip_count}")
            save_folder = f"/data/yongka/analogy/spatial/error_ana/property"
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            save_path = f"{save_folder}/{p[0]}_{p[1]}.jsonl"
            with open(save_path, 'w') as f:
                for item in sampled_data[:50]:
                    f.write(json.dumps(item) + '\n')