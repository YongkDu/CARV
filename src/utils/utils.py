"""
Shared utilities for the CARV benchmark.

Provides model loading, prompt loading, conversation building, JSON extraction,
semantic correctness checking, and GPT-4o image-based evaluation used by the
inference and evaluation scripts.
"""
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM
import json
import os
import re
import random
from tqdm import tqdm
import time
import yaml
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import base64

PROMPT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../prompts/analyze.yaml")

# Directory containing API key JSON config files.
# Override by setting the CARV_CONFIG_DIR environment variable.
_DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../configs")
CONFIG_DIR = os.environ.get("CARV_CONFIG_DIR", _DEFAULT_CONFIG_DIR)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_model(model_name):
    def _cfg(filename):
        return json.load(open(os.path.join(CONFIG_DIR, filename)))

    if model_name == "qwen":
        api_config = _cfg("qwen_vl_api_key_config.json")
        model_id = api_config["model_id"]
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor, api_config
    elif model_name == "qwen32b":
        api_config = _cfg("qwen32b_vl_api_key_config.json")
        model_id = api_config["model_id"]
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor, api_config
    elif model_name == "llama":
        api_config = _cfg("llama_api_key_config.json")
        model_id = api_config["model_id"]
        model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor, api_config
    elif model_name == "gemini2.5pro":
        api_config = _cfg("geminipro_api_key_config.json")
        return None, None, api_config
    elif model_name == "gemini2.5flash":
        api_config = _cfg("geminiflash_api_key_config.json")
        return None, None, api_config
    elif model_name == "gpt":
        api_config = _cfg("openai_api_key_config.json")
        return None, None, api_config
    elif model_name == "gpt5" or model_name == "gpt5t":
        api_config = _cfg("openai_api_key_config2.json")
        return None, None, api_config
    elif model_name == "o1":
        api_config = _cfg("openai_api_key_config2.json")
        return None, None, api_config
    elif model_name == "llava":
        api_config = _cfg("llava_api_key_config.json")
        model_id = api_config["model_id"]
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor, api_config
    elif model_name == "qwen3":
        api_config = _cfg("qwen3_vl_api_key_config.json")
        model_id = api_config["model_id"]
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor, api_config
    elif model_name == "qwen3-thinking":
        api_config = _cfg("qwen3_thinking_api_key_config.json")
        model_id = api_config["model_id"]
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return model, processor, api_config
    else:
        raise ValueError(f"Model {model_name} not supported.")

def get_prompt(prompt_root, prompt_type):
    """Load a named prompt from a YAML prompt file."""
    with open(prompt_root, "r") as f:
        prompt_template = yaml.safe_load(f)
    prompt = prompt_template["prompts"][prompt_type]["text"]
    return prompt

def build_conversation(context_path, task_description):
    context_images = [Image.open(p) for p in context_path]
    if len(context_images) == 5:
        prompt = [
            {"type": "text", "text": task_description},
            {"type": "text", "text": "Image I1"}, {"type": "image", "image": context_images[0]},
            {"type": "text", "text": "Image I2"}, {"type": "image", "image": context_images[1]},
            {"type": "text", "text": "Image I3"}, {"type": "image", "image": context_images[2]},
            {"type": "text", "text": "Image I4"}, {"type": "image", "image": context_images[3]},
            {"type": "text", "text": "Image I5"}, {"type": "image", "image": context_images[4]},
        ]
    elif len(context_images) == 3:
        prompt = [
            {"type": "text", "text": task_description},
            {"type": "text", "text": "Image I1"}, {"type": "image", "image": context_images[0]},
            {"type": "text", "text": "Image I2"}, {"type": "image", "image": context_images[1]},
            {"type": "text", "text": "Image I3"}, {"type": "image", "image": context_images[2]},
        ]
    return prompt, context_images

def extract_transformation(s):
    # find the substring after the last "Transformation T:"
    target_string = "Transformation T"
    start_idx = s.rfind(target_string)
    if start_idx == -1:
        return ""
    transformation = s[start_idx:].strip()
    return transformation

def extract_json(s):
    # find substring between ```json and ```
    pattern = r"json(.*?)```"
    matches = re.findall(pattern, s, re.DOTALL)
    s = matches[-1].strip() if matches else s
    start = s.rfind("{")
    end = s.rfind("}")
    if end == -1:
        s = s + "}"
        end = s.find("}")
    data_dict = None
    if start != -1 and end != -1:
        llm_response_string = s[start:end+1]
        if '//' in llm_response_string:
            # remove comments starting with //
            llm_response_string = re.sub(r"//.*?\n", "", llm_response_string)
        try:
            data_dict = json.loads(llm_response_string)
        except json.JSONDecodeError:
            key_to_fix = '"reasoning_summary":'
            bad_json_string = llm_response_string
            key_index = bad_json_string.find(key_to_fix)

            if key_index != -1:
                value_start_index = key_index + len(key_to_fix)
                last_brace_index = bad_json_string.rfind('}')

                if last_brace_index == -1 or last_brace_index <= value_start_index:
                    return None
                else:
                    text_to_quote = bad_json_string[value_start_index:last_brace_index].strip()
                    correctly_quoted_value = json.dumps(text_to_quote)
                    fixed_json_string = (
                        bad_json_string[:value_start_index] +
                        " " +
                        correctly_quoted_value +
                        bad_json_string[last_brace_index:]
                    )
                    try:
                        data_dict = json.loads(fixed_json_string)
                    except json.JSONDecodeError:
                        return None
            else:
                return None
    return data_dict


def check_correctness(model, answer, label, threshold1=0.6, threshold2=0.9, threshold3=0.7):
    """
    Compute thresholded semantic accuracy (TSA) for a predicted answer triple.

    Uses cosine similarity between SBERT embeddings of each component (subject,
    predicate, object). Returns 1 if all three components exceed their respective
    thresholds, else 0.
    """
    answer_list = [answer[ans] for ans in answer.keys()]
    if 'of' in label:
        label = label.replace('_of', '')
    if 'double' in label:
        label = label.replace('double', 'two')
    if 'single' in label:
        label = label.replace('single', 'one')
    label_list = label.split('_')

    if 'wood' in label_list and ('brown' in answer_list[0] or 'beige' in answer_list[0] or 'wood' in answer_list[0]):
        answer_list[0] = 'wood'
    if 'metal' in label_list and ('grey' in answer_list[0] or 'gray' in answer_list[0] or 'silver' in answer_list[0]):
        answer_list[0] = 'metal'
    if 'cap' in label_list and 'hat' in answer_list[0]:
        answer_list[0] = 'cap'

    all_parts = label_list + answer_list
    embeddings = model.encode(all_parts, convert_to_tensor=True)

    sim_s = max(0, util.cos_sim(embeddings[0], embeddings[3]).item())
    sim_p = max(0, util.cos_sim(embeddings[1], embeddings[4]).item())
    sim_o = max(0, util.cos_sim(embeddings[2], embeddings[5]).item())

    pass_s = sim_s >= threshold1
    pass_p = sim_p >= threshold2
    pass_o = sim_o >= threshold3

    if pass_s and pass_p and pass_o:
        tsa_score = 1
    else:
        tsa_score = 0
    del embeddings
    return tsa_score


def get_response_from_gpt_img(task_description, context_path):
    api_config = json.load(open(os.path.join(CONFIG_DIR, "openai_api_key_config_acc.json")))
    content_parts = []
    content_parts.append({
        "type": "text",
        "text": task_description
    })
    with open(context_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    if base64_image:
        mime_type = "image/jpeg"
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })
    try:
        client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content_parts,
                }
            ],
            temperature=api_config.get("temperature", 0),
            max_tokens=api_config.get("max_tokens", 1024)
        )
        output = response.choices[0].message.content
    except Exception as e:
        print(f"GPT Error: {e}")
    return output

def check_correctness_gpt(item, task, img_root):

    if item["answer"] is None:
        item["correctness_analysis"] = {'correctness': None, 'reason': 'No answer provided'}
        return item
    caption = item["answer"].get('caption', None)
    if caption is None:
        item["correctness_analysis"] = {'correctness': None, 'reason': 'No caption provided'}
        return item
    prompt = get_prompt(PROMPT_ROOT, "general")
    prompt = prompt.format(caption=caption, label_caption=item['label'])
    img_path = f"{img_root}/{item['label']}.jpeg"
    evaluation = get_response_from_gpt_img(prompt, img_path)
    try:
        result = extract_json(evaluation)
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        result = {'correctness': None, 'reason': 'Failed in analysis'}

    item["correctness_analysis"] = result
    a=1
    return item
