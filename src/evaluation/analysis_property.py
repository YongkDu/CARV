"""
Property-level error analysis.

Runs GPT-4o diagnosis on a targeted subset of responses from error_ana/property/
that are grouped by property combination (e.g. subject_position). Used to
examine how model failure rates vary across different property types, supporting
the property-level breakdown in the paper.

Usage:
    Set model_response_path to error_ana/property/data/ in __main__, then:
        python analysis_3step_property.py
"""
import json
import os
import base64
from openai import OpenAI
import openai
from google import genai
from google.genai.types import Content, Part
from utils import extract_json, read_jsonl, get_prompt

PROMPT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../prompts/analyze.yaml")

def get_response_from_gpt(api_config, prompt):
    if api_config["model"] == "gpt-4o":
        client = OpenAI(api_key=api_config["api_key"])
        response = client.chat.completions.create(
            model=api_config["model"],
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "temperature": 0
                }
            ],
            max_tokens=1024
        )
        output = response.choices[0].message.content
    elif api_config["model"] == "o1" or  api_config["model"] == "gpt-5-mini" or api_config["model"] == "gpt-5":
        try:
            client = OpenAI(api_key=api_config["api_key"])
            response = client.chat.completions.create(
                model=api_config["model"],  # Or "o1-mini" for the smaller version
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            output = response.choices[0].message.content
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
    elif api_config["model"] == "gpt-5.1":
        content_parts = []
        content_parts.append({
            "type": "input_text",
            "text": prompt
        })
        try:
            client = OpenAI(api_key=api_config["api_key"])
            response = client.responses.create(
                model=api_config["model"],
                input=[{
                        "role": "user",
                        "content": content_parts,
                        }],
                reasoning={"effort": "low"},
                text={"verbosity": "medium"},
                max_output_tokens=api_config["max_tokens"]
            )
            output = response.output_text   
        except Exception as e:
            print(f"GPT Error: {e}")
            return None
        
    return output

def get_response_from_gpt_img(api_config, task_description, context_path):
    if api_config["model"] == "gpt-4o":
        content_parts = []
        content_parts.append({
            "type": "text",
            "text": task_description
        })
        image_paths = context_path
        for index, path in enumerate(image_paths, start=1):
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            if base64_image:
                mime_type = "image/jpeg"
                content_parts.append({
                    "type": "text",
                    "text": f"I{index}"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
        try:
            client = OpenAI(api_key=api_config["api_key"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": content_parts,
                        "temperature": api_config["temperature"],
                        "seed": api_config.get("seed", None)
                    }
                ],
                max_tokens=api_config["max_tokens"]
            )
            output = response.choices[0].message.content
        except Exception as e:
            print(f"GPT Error: {e}")
        
    elif api_config["model"] == "gpt-5.1":
        content_parts = []
        
        content_parts.append({
            "type": "input_text",
            "text": task_description
        })

        image_paths = context_path
        for index, path in enumerate(image_paths, start=1):
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            if base64_image:
                mime_type = "image/jpeg"
                content_parts.append({
                    "type": "input_text",
                    "text": f"Image {index}"
                })
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{base64_image}"
                })

        try:
            client = OpenAI(api_key=api_config["api_key"])
            response = client.responses.create(
                model="gpt-5.1",
                input=[{
                        "role": "user",
                        "content": content_parts
                        }],
                reasoning={"effort": "low"},
                text={"verbosity": "medium"},
                max_output_tokens=api_config["max_tokens"]
            )
            output = response.output_text   
        except Exception as e:
            print(f"GPT Error: {e}")
            return None
    return output

# def get_response_from_gemini(api_config, prompt):

def get_transformation(s):
    # find the substring after the last "Transformation T:"
    target_string = "Transformation T"
    start_idx = s.rfind(target_string)
    if start_idx == -1:
        return ""
    transformation = s[start_idx:].strip()
    return transformation

def check_changed_property(changed_property):
    return_string = ""
    for key,v in changed_property.items():
        return_string += f"{v[0]} to {v[1]}; "
    return return_string

def extract_transformation(image1, image2, category):
    # split the image names by underscore, i.e. "subject_right_of_object", get subject, position, object
    if '_of' in image1:
        image1 = image1.replace('_of', '')
    if '_of' in image2:
        image2 = image2.replace('_of', '')
    parts1 = image1.split("_")
    parts2 = image2.split("_")
    if "mix" in category or "large" in category:
        subject1, number1, position1, object1, color1 = parts1[0], parts1[1], parts1[2], parts1[3], parts1[4]
        subject2, number2, position2, object2, color2 = parts2[0], parts2[1], parts2[2], parts2[3], parts2[4]
    elif "subject" in category:
        subject1, position1, object1 = parts1[0], parts1[1], parts1[-1]
        subject2, position2, object2 = parts2[0], parts2[1], parts2[-1]
        color1, number1 = None, None
        color2, number2 = None, None
    elif "color" in category:
        color1, number1, object1 = parts1[0], parts1[1], parts1[-1]
        color2, number2, object2 = parts2[0], parts2[1], parts2[-1]
        subject1, position1 = None, None
        subject2, position2 = None, None
    subject_change = ""
    number_change = ""
    position_change = ""
    object_change = ""
    color_change = ""
    # nl_description = ""
    changed_properties = {}
    if subject1 != subject2:
        changed_properties["subject"] = (subject1, subject2)
        subject_change = f"subject change from {subject1} to {subject2}. "
    if number1 != number2:
        changed_properties["number"] = (number1, number2)
        number_change = f"number change from {number1} to {number2}. "
    if position1 != position2:
        changed_properties["position"] = (position1, position2)
        position_change = f"position change from {position1} to {position2}. "
    if object1 != object2:
        changed_properties["object"] = (object1, object2)
        object_change = f"object change from {object1} to {object2}."
    if color1 != color2:
        changed_properties["color"] = (color1, color2)
        color_change = f"color change from {color1} to {color2}. "
    return changed_properties, subject_change + number_change + position_change + object_change + color_change


def analyze_with_gpt(response_file, label_file, analyze_file, api_config, operation, img_root):
    # read the label file
    # with open(label_file, "r") as f:
    #     label = json.load(f)
    # read the model response file
    wrong_data = ['two_toilet-roll_left_of_blue_chair', 'two_toilet-roll_left_of_red_chair', 'two_toilet-roll_left_of_wood_chair',]
    responses = read_jsonl(response_file)
    # read analysis file if exists
    analysis_result = read_jsonl(analyze_file) if os.path.exists(analyze_file) else []
    start_index = len(analysis_result)
    result2save = []
    for i,item in enumerate(responses):
        if i < start_index:
            result2save.append(analysis_result[i])
            print(f"***Skipped {i}/{len(responses)}***")
            continue
        # if item['correctness_analysis']['correctness'] == True:
        #     result = {'failure_stage': 0, 'reasoning_summary': 'The model answer is correct.'}
        # else:
        prompt = get_prompt(PROMPT_ROOT, "overall")
        response = item["response"]
        context_path = [f"{img_root}/{p}.jpeg" for p in item["context_images"][:-1]]
        captions = item["context_images"]
        _, label1 = extract_transformation(item["context_images"][0], item["context_images"][1], "mix")
        _, label2 = extract_transformation(item["context_images"][2], item["context_images"][3], "mix")
        _, target_label = extract_transformation(item["context_images"][-1], item["label"], "mix")
        analyze_instruction = prompt.format(
            response=response,
            caption1=captions[0],
            caption2=captions[1],
            caption3=captions[2],
            caption4=captions[3],
            ground_truth_T1=label1,
            ground_truth_T2=label2,
            target_transformation=target_label,
            caption5=captions[4]
        )
        
        analysis = get_response_from_gpt(api_config, analyze_instruction)
        # print(analysis)
        print(analysis)
    
        result = extract_json(analysis)
        if result is None:
            result = {'failure_stage': None, 'reasoning_summary': analysis}
        print(f"***Analyzed {i}/{len(responses)}***")
        result2save.append(result)
        # count += 1
        a=1
        # save current output to jsonl file
    with open(analyze_file, "w") as f:
        for res in result2save:
            f.write(json.dumps(res) + "\n")
def analyze_compositional_files(model_response_path, api_config, tasks, models, difficulties, operation, data_size):
    analyze_folder = f"/data/yongka/analogy/spatial/error_ana/property/ana"
    img_root = "/data/yongka/analogy/spatial/data/mix"
    for combi in ['subject_position']: #, 'object_color', 'subject_position'
        response_file = f"{model_response_path}/{combi}.jsonl"
        analyze_file = f"{analyze_folder}/{combi}.jsonl"
        analyze_with_gpt(response_file, None, analyze_file, api_config, None, img_root)

if __name__ == "__main__":
    api_config = json.load(open("/data/yongka/analogy/configs/openai_api_key_config_eval.json"))
    # api_config = json.load(open("/data/yongka/analogy/configs/geminiflash_api_key_config_eval.json"))
    # output_type = ["base"]
    task_category = ["large"] #
    operations = ["intersection", "union"]
    data_size = 500
    models = ['gemini2.5pro']
    difficulties = ['hard']
    # for ot in output_type:
    model_response_path = f"/data/yongka/analogy/spatial/error_ana/property/data"
    # analysis_path = f"/data/yongka/analogy/spatial/analysis_{ot}"
    analyze_compositional_files(model_response_path, api_config, task_category, models, difficulties, operations, data_size)