"""
Accuracy computation for 1-step analysis outputs.

Reads analysis JSONL files from analysis_1step/ folders and computes
application-correctness accuracy per model, relationship type, and difficulty
level for the single-step baseline evaluation.
"""
import matplotlib.pyplot as plt
import numpy as np
import json

# read jsonl file
def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_json(s):
    start = s.find("{")
    end = s.find("}")
    if end == -1:
        end = len(s) -1
    if start != -1 and end != -1:
        llm_response_string = s[start:end+1]
        # print(llm_response_string)
        try:
            data_dict = json.loads(llm_response_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    return data_dict

def acc_step_by_step(analyze_file, output_file):
    data_analyze = load_jsonl(analyze_file)
    data_output = load_jsonl(output_file)
    results = []
    mistaking_correct = 0
    data_analyze_new = []
    correct_n = 0
    for i, item in enumerate(data_analyze):
        # analysis_stepone = extract_json(item["analysis_stepone"])
        # analysis_steptwo = extract_json(item["analysis_steptwo"])
        analysis = extract_json(item.get("analysis", "{}"))
        # print(analysis_stepone['grounding_correctness'],'\n',analysis_stepone['reasoning_summary'])
        # print(analysis_steptwo['operation_correctness'],'\n',analysis_steptwo['reasoning_summary'])
        # print("-----")
        if analysis is None:
            continue
        # merge two analysis dicts
        data_dict = {}
        # data_dict.update(analysis_stepone)
        # data_dict.update(analysis_steptwo)
        # data_dict.update(analysis_stepthree)
        # if data_output[i]['correct'] == True:
        #     data_dict['task_correctness'] = True
        # else:
        #     data_dict['task_correctness'] = False
        results.append(data_dict)
        if analysis['application_correctness'] == True:
            correct_n += 1
    #     # if data_output[i]['correct'] == 1:
    #     #     data_dict['task_correctness'] = True
    #     # else:
    #     #     data_dict['task_correctness'] = False
    #     results.append(data_dict)
    
    # # compute the accuracy for each step
    # correct_counts = [0,0,0]
    # for item in results:
    #     if item['grounding_correctness']:
    #         correct_counts[0] += 1
    #     if item['operation_correctness']:
    #         correct_counts[1] += 1
    #     if item['task_correctness']:
    #         correct_counts[2] += 1
    # acc_each_step = []
    # # round to two decimal places
    # for count in correct_counts:
    #     acc = count / len(results)
    #     acc = round(acc, 2)
    #     acc_each_step.append(acc)
    acc = correct_n / len(results)
    acc = round(acc, 2)
    return acc


def stat_analyze_files(models, task, output_type):
    difficulties = ["easy", "hard"]
    task_category = [task]
    data_size = 100
    dict_acc = {}
    for ot in output_type:
        analysis_path = f"/data/yongka/analogy/spatial/analysis_{ot}"
        output_path = f"/data/yongka/analogy/spatial/output_{ot}"
        relationship = ["union", "difference", "intersection"]
        for task in task_category:
            for rel in relationship:
                analysis_folder = analysis_path + f"/{task}/{rel}"
                output_folder = output_path + f"/{task}/{rel}"
                for difficulty in difficulties:
                    for model in models:
                        analyze_file = f"{analysis_folder}/{model}_{difficulty}_{data_size}.jsonl"
                        output_file = f"{output_folder}/{model}_{difficulty}_{data_size}.jsonl"
                        acc = acc_step_by_step(analyze_file, output_file)
                        if f"{task}_{rel}_{difficulty}_{model}" not in dict_acc:
                            dict_acc[f"{task}_{rel}_{difficulty}_{model}"] = []
                        dict_acc[f"{task}_{rel}_{difficulty}_{model}"].append(acc)
            # analyze single files
            # analysis_folder = analysis_path + f"/{task}/single"
            # output_folder = output_path + f"/{task}/single"
            # if task == "subject":
            #     properties = ["subject", "position"]
            # else:
            #     properties = ["color", "number"]
            # for p in properties:
            #     for model in models:
            #         analyze_file = f"{analysis_folder}/{model}_{p}_{data_size}.jsonl"
            #         output_file = f"{output_folder}/{model}_{p}_{data_size}.jsonl"
            #         acc = acc_step_by_step(analyze_file, output_file)
            #         if f"{task}_{p}_single_{model}" not in dict_acc_s:
            #             dict_acc_s[f"{task}_{p}_single_{model}"] = []
            #         dict_acc_s[f"{task}_{p}_single_{model}"].append(acc)

    # print the dict_acc
    for key in dict_acc:
        print(f"{key}: {dict_acc[key]}")
    return dict_acc

def transform_for_plotting(dict_acc):
    """
    Transform dict_acc to a list of lists for plotting.
    Each sublist corresponds to a step, and contains accuracy values for different categories.
    """
    # transform dict_acc to a list of lists for plotting
    data = []
    # put data in dict_acc to data, ith element of each dict_acc[key]
    for key in dict_acc:
        for i in range(len(dict_acc[key])):
            if i >= len(data):
                data.append([])
            data[i].append(dict_acc[key][i])
    # flatten each sublist in data
    flattened_data = []
    for i in range(len(data)):
        sublist = data[i]
        for item in sublist:
            flattened_data.append(item)
    return flattened_data

model = "gpt"
task = "subject"
prompt_type = ["1step"]
dict_acc = stat_analyze_files([model], task, prompt_type)
# data = transform_for_plotting(dict_acc)
# print(data)
