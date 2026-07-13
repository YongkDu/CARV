"""
Convert intersection datasets to difference datasets by recomputing labels.

Reads intersection task definition JSONs and produces corresponding difference
task JSONs by inverting the set membership logic for each item's label.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

ATTR_KEYS = ["number", "subject", "position", "color", "object"]
POSITION_TOKEN_PATTERNS = [
    ("left", "of"),
    ("right", "of"),
    ("on",),
    ("under",),
]


def parse_image_string(image_str: str) -> Dict[str, str]:
    parts = image_str.split("_")
    if len(parts) < 5:
        raise ValueError(f"Invalid image string (expected >=5 underscore-separated parts): {image_str}")

    number = parts[0]
    color = parts[-2]
    obj = parts[-1]

    middle_tokens = parts[1:-2]
    if not middle_tokens:
        raise ValueError(f"Invalid image string (missing subject/position segment): {image_str}")

    matched_position_tokens = None
    for pattern in POSITION_TOKEN_PATTERNS:
        pattern_len = len(pattern)
        if len(middle_tokens) >= pattern_len and tuple(middle_tokens[-pattern_len:]) == pattern:
            matched_position_tokens = pattern
            break

    if matched_position_tokens is None:
        raise ValueError(f"Invalid image string (cannot infer position): {image_str}")

    position_len = len(matched_position_tokens)
    subject_tokens = middle_tokens[:-position_len]
    if not subject_tokens:
        raise ValueError(f"Invalid image string (missing subject): {image_str}")

    subject = "_".join(subject_tokens)
    position = "_".join(matched_position_tokens)

    return {
        "number": number,
        "subject": subject,
        "position": position,
        "color": color,
        "object": obj,
    }


def to_image_string(attrs: Dict[str, str]) -> str:
    return "_".join([attrs[key] for key in ATTR_KEYS])


def get_transformation(source: Dict[str, str], target: Dict[str, str]) -> Dict[str, str]:
    return {k: target[k] for k in ATTR_KEYS if source[k] != target[k]}


def remove_common_transformations(t1: Dict[str, str], t2: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in t1.items() if not (k in t2 and t2[k] == v)}


def derive_difference_label(context_image: List[str]) -> str:
    if len(context_image) < 5:
        raise ValueError(f"context_image needs at least 5 elements, got {len(context_image)}")

    image1 = parse_image_string(context_image[0])
    image2 = parse_image_string(context_image[1])
    image3 = parse_image_string(context_image[2])
    image4 = parse_image_string(context_image[3])
    image5 = parse_image_string(context_image[4])

    t1 = get_transformation(image1, image2)
    t2 = get_transformation(image3, image4)

    t1_remaining = remove_common_transformations(t1, t2)

    output = dict(image5)
    output.update(t1_remaining)
    return to_image_string(output)


def convert_record(record: Dict) -> Dict:
    if "context_image" not in record:
        raise KeyError("Missing key 'context_image' in record")

    updated = dict(record)
    updated["label"] = derive_difference_label(record["context_image"])
    return updated


def convert_file(input_path: str, output_path: str) -> Tuple[int, int]:
    with open(input_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in JSON file: {input_path}")

    converted = [convert_record(item) for item in data]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=4)

    return len(data), len(converted)


def find_intersection_json_files(data_root: str) -> List[str]:
    matches = []
    for root, _, files in os.walk(data_root):
        if os.path.basename(root) != "intersection":
            continue
        for filename in files:
            if filename.endswith(".json"):
                matches.append(os.path.join(root, filename))
    return sorted(matches)


def get_output_path(intersection_file_path: str) -> str:
    intersection_dir = os.path.dirname(intersection_file_path)
    parent_dir = os.path.dirname(intersection_dir)
    difference_dir = os.path.join(parent_dir, "difference")
    return os.path.join(difference_dir, os.path.basename(intersection_file_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert intersection datasets into difference datasets by recomputing labels from context_image."
    )
    parser.add_argument(
        "--data-root",
        default="/data/yongka/analogy/spatial/data_path_all/mix",
        help="Root folder to scan for */intersection/*.json",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional single intersection json file to process",
    )
    args = parser.parse_args()

    if args.input_file:
        input_files = [args.input_file]
    else:
        input_files = find_intersection_json_files(args.data_root)

    if not input_files:
        print("No intersection JSON files found.")
        return

    total_records = 0
    for input_path in input_files:
        output_path = get_output_path(input_path)
        n_in, _ = convert_file(input_path, output_path)
        total_records += n_in
        print(f"Converted: {input_path} -> {output_path} ({n_in} records)")

    print(f"Done. Converted {len(input_files)} file(s), {total_records} record(s).")


if __name__ == "__main__":
    main()
