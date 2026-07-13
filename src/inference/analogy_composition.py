"""
Main inference script for the CARV benchmark.

Runs VLMs on mix, large, or single tasks using the specified prompting strategy
(direct or step_by_step), evaluates each response with GPT-4o, and saves
per-item results to JSONL files under outputs/.

Supported models: gemini2.5pro, gemini2.5flash, gpt, gpt5/gpt5t, llama, qwen,
                  qwen32b, qwen3, qwen3-thinking, llava, o1

Usage:
    python analogy_composition.py --task mix --prompt step_by_step \\
        --model gemini2.5flash --number 500
"""
import base64
import json
import os
import re
import random
import sys
import time
import yaml
import argparse
from tqdm import tqdm
from google import genai
from google.genai.types import Content, Part
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../utils"))
from utils import load_model, get_prompt, build_conversation, extract_json, check_correctness_gpt, read_jsonl

prompt_dict = {
    'intersection': "INTERSECTION: The Target Transformation should be the transformation(s) that are common to BOTH T1 AND T2.",
    'union': "UNION: The Target Transformation should combine the transformation(s) that appear in T1 and T2.",
    'difference': "DIFFERENCE: The Target Transformation should be the transformation(s) that appear in T1 but NOT in T2."
    }

def analogy_reasoning_step(context, api_config, model, processor, task_description):
    context_path = [f"{IMAGE_ROOT}/{p}.jpeg" for p in context] if context is not None else []
    if api_config["model"] == "qwen" or api_config["model"] == "llama" or api_config["model"] == "llava":
        conversation, image_list = build_conversation(context_path, task_description)
        messages = [{"role": "user", "content": conversation}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            images=image_list,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        generation_args = {
            "max_new_tokens": api_config["max_tokens"],
            "temperature": api_config["temperature"],
            "do_sample": True,
            "top_p": api_config["top_p"],
        }
        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args
        )
        try:
            output = processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        except Exception as e:
            print(f"{api_config['model']} Error: {e}")
            return None
    elif api_config["model"] == "gemini":
        image_paths = context_path
        contents = [task_description]

        for index, path in enumerate(image_paths, start=1):
            image_data = open(path, "rb").read()
            contents.append(f"Image {index}")
            contents.append(
                Part.from_bytes(
                    data=image_data,
                    mime_type="image/jpeg"
                )
            )
        max_retries = 5
        for j in range(max_retries):
            try:
                client = genai.Client(api_key=api_config["api_key"])
                response = client.models.generate_content(
                    model=api_config["model_id"],
                    contents=contents
                )
                break
            except Exception as e:
                if j == max_retries - 1:
                    raise e
                base_wait = min(3 ** j, 30)
                wait_time = base_wait + random.uniform(0, 1)
                time.sleep(wait_time)
        try:
            output = response.text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return None
    elif api_config["model"] == "gpt-4o":
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
                        "content": content_parts
                    }
                ],
                max_tokens=api_config["max_tokens"]
            )
            output = response.choices[0].message.content
        except Exception as e:
            print(f"GPT Error: {e}")
            return None
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
                client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
                response = client.responses.create(
                    model="gpt-5.1",
                    input=[{
                            "role": "user",
                            "content": content_parts
                            }],
                    reasoning={"effort": "low"},
                    text={
                            "verbosity": "medium"
                        },
                    max_output_tokens=api_config["max_tokens"]
                )
                output = response.output_text
            except Exception as e:
                print(f"GPT Error: {e}")
                return None
    print(f"Generated output:\n {output}")
    return output

def main(difficulty, operation, args, model, processor, api_config):
    with open(f"{DATA_ROOT}/{operation}/{difficulty}_{args.number}.json", "r") as f:
        sampled_data = json.load(f)
    output_path = f"{OUTPUT_ROOT}/{operation}/{args.model}_{difficulty}_{args.number}.jsonl"

    if os.path.exists(output_path):
        os.remove(output_path)
        exist_len = 0
        start_idx = 0
    else:
        exist_len = 0
        start_idx = 0
    task_descriptions = get_prompt(PROMPT_ROOT, args.prompt)
    task_descriptions = task_descriptions.format(operation=operation.upper(), description=prompt_dict[operation])
    if args.model == "gpt5" or args.model == "qwen":
        task_descriptions += "\nYou MUST provide your reasoning process before giving the final answer."
    print(f"***Loaded {args.model} model for {operation}_{difficulty}_{args.number}.***")
    evaluated = 0
    for i,d in tqdm(enumerate(sampled_data)):
        label = d['label']
        max_attempts = 3
        response = None
        answer = None
        for t in range(max_attempts):
            response = analogy_reasoning_step(d["context_image"], api_config, model, processor, task_descriptions)
            if response is None:
                continue
            answer = extract_json(response)
            if answer is not None:
                break
        cur_data = {"response": response, "answer": answer, "context_images": d["context_image"], "label": label}
        cur_data = check_correctness_gpt(cur_data, args.task, IMAGE_ROOT)
        print(f"Current Data Correctness Analysis: {cur_data['correctness_analysis']}")
        if not os.path.exists(f"{OUTPUT_ROOT}/{operation}"):
            os.makedirs(f"{OUTPUT_ROOT}/{operation}")
        with open(f"{OUTPUT_ROOT}/{operation}/{args.model}_{difficulty}_{args.number}.jsonl", "a") as f:
            f.write(json.dumps(cur_data) + "\n")
            print(f"Saved {start_idx+i} result to jsonl file.")
        evaluated += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mix")
    parser.add_argument("--prompt", type=str, default="step_by_step")
    parser.add_argument("--number", type=int, default=500)
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="gpt5")
    args = parser.parse_args()
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if args.task == 'single':
        IMAGE_ROOT = f"/data/yongka/analogy/spatial/data/mix"
    else:
        IMAGE_ROOT = f"/data/yongka/analogy/spatial/data/{args.task}"
    DATA_ROOT = f"/data/yongka/analogy/spatial/data_path_all/{args.task}"
    prompt_file = "mix.yaml" if args.task == "large" else f"{args.task}.yaml"
    PROMPT_ROOT = os.path.join(PROJECT_ROOT, "prompts", prompt_file)
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", args.prompt, args.task)

    print(f"***Compositional Task type: {args.task}***")
    print(f"***Using {args.prompt} prompts***")
    model, processor, api_config = load_model(args.model)
    difficulty_list = ["hard"]
    operations = ["union", "intersection"]

    for difficulty in difficulty_list:
        for operation in operations:
            main(difficulty, operation, args, model, processor, api_config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
