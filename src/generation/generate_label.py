"""
Generate step-by-step labels (T1, T2, target_label) for all task JSON files.

Reads task definition JSONs and computes the ground-truth transformation labels
for each item's three steps, writing enriched label files used by the evaluation
pipeline.
"""
PROJ_ROOT = "/data/yongka/analogy/spatial"
CATEGORY = "subject"
DATA_PATH = f"{PROJ_ROOT}/data_path_all/{CATEGORY}"
LABEL_PATH = f"{PROJ_ROOT}/data_labels_all/{CATEGORY}"
import json
import os
import random

if CATEGORY == "subject":
    PROPERTY_1 = "subject"
    PROPERTY_2 = "position"
    PROPERTY_3 = "object"
elif CATEGORY == "color":
    PROPERTY_1 = "color"
    PROPERTY_2 = "number"
    PROPERTY_3 = "object"

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def extract_transformation(image1, image2):
    # split the image names by underscore, i.e. "subject_right_of_object", get subject, position, object
    parts1 = image1.split("_")
    parts2 = image2.split("_")
    subject1, position1, object1 = parts1[0], parts1[1], parts1[-1]
    subject2, position2, object2 = parts2[0], parts2[1], parts2[-1]
    subject_change = ""
    position_change = ""
    object_change = ""
    nl_description = ""
    changed_properties = {}
    if subject1 != subject2:
        changed_properties[PROPERTY_1] = (subject1, subject2)
        subject_change = f"{PROPERTY_1} change from {subject1} to {subject2}. "
    if position1 != position2:
        changed_properties[PROPERTY_2] = (position1, position2)
        position_change = f"{PROPERTY_2} change from {position1} to {position2}. "
    if object1 != object2:
        changed_properties[PROPERTY_3] = (object1, object2)
        object_change = f"{PROPERTY_3} change from {object1} to {object2}."
    return changed_properties, subject_change + position_change + object_change

def step_by_step_label(item):
    # read context images and options
    context = item["context_image"]
    changed_properties1,context_label1 = extract_transformation(context[0], context[1])
    context_label2 = None
    if len(context)>3: # for composition analogy
        changed_properties2, context_label2 = extract_transformation(context[2], context[3])
    _, label = extract_transformation(context[-1], item["label"])
    return changed_properties1, context_label1, changed_properties2, context_label2, label

def generate_labels():
    # read all json files in subfolder of DATA_PATH
    all_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    for folder in all_folders:
        folder_path = os.path.join(DATA_PATH, folder)
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        for file in all_files:
            file_path = os.path.join(folder_path, file)
            # generate step-by-step labels for each json file
            data = read_json(file_path)
            labeled_data = []
            for item in data:
                changed_properties1, context_label1, changed_properties2, context_label2, label = step_by_step_label(item)
                # add labels to item
                # item['transformations'] = labels
                if context_label2 is not None: # for composition analogy
                    labeled_data.append({
                        "changed_properties1": changed_properties1,
                        "changed_properties2": changed_properties2,
                        "context_label1": context_label1,
                        "context_label2": context_label2,
                        "target_label": label,
                        # "wrong_subject_label": wrong_sub_label,
                        # "wrong_position_label": wrong_pos_label
                    })
                else: # for single analogy
                    labeled_data.append({
                        "context_label": context_label1,
                        "target_label": label,
                        # "wrong_subject_label": wrong_sub_label,
                        # "wrong_position_label": wrong_pos_label
                    })
            # save labeled_data to LABEL_PATH with same folder structure
            label_folder = os.path.join(LABEL_PATH, folder)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            label_file_path = os.path.join(label_folder, file)
            with open(label_file_path, "w") as f:
                json.dump(labeled_data, f, indent=4)
            
            # save the updated data with transformations to DATA_PATH
            # with open(file_path, "w") as f:
            #     json.dump(data, f, indent=4)
            print(f"Processed {file_path}")
            # print(f"Saved labels to {label_file_path}")
    return

if __name__ == "__main__":
    generate_labels()