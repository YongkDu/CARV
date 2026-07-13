"""
Extract transformation labels for single-step evaluation tasks.

Processes image pairs from the mix dataset to identify single atomic
transformations between source and target images, creating single-step
task definition files under data_path_all/single/.
"""
PROJ_ROOT = "/data/yongka/analogy/spatial"
DATA_PATH = f"{PROJ_ROOT}/data_path_all"
import json
import os
import random
import copy

def extract_transformation(context_image1, context_image2, context_image3):
    image1 = copy.deepcopy(context_image1)
    image2 = copy.deepcopy(context_image2)
    image3 = copy.deepcopy(context_image3)
    # split the image names by underscore, i.e. "subject_right_of_object", get subject, position, object
    if '_of' in image1:
        image1 = image1.replace('_of', '')
    if '_of' in image2:
        image2 = image2.replace('_of', '')
    if '_of' in image3:
        image3 = image3.replace('_of', '')
    parts1 = image1.split("_")
    parts2 = image2.split("_")
    parts3 = image3.split("_")
    number1, subject1, position1, color1, object1 = parts1[0], parts1[1], parts1[2], parts1[3], parts1[4]
    number2, subject2, position2, color2, object2 = parts2[0], parts2[1], parts2[2], parts2[3], parts2[4]
    number3, subject3, position3, color3, object3 = parts3[0], parts3[1], parts3[2], parts3[3], parts3[4]
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
        if subject3 == subject1:
            subject3 = subject2
    if number1 != number2:
        changed_properties["number"] = (number1, number2)
        number_change = f"number change from {number1} to {number2}. "
        if number3 == number1:
            number3 = number2
    if position1 != position2:
        changed_properties["position"] = (position1, position2)
        position_change = f"position change from {position1} to {position2}. "
        if position3 == position1:
            position3 = position2
    if object1 != object2:
        changed_properties["object"] = (object1, object2)
        object_change = f"object change from {object1} to {object2}."
        if object3 == object1:
            object3 = object2
    if color1 != color2:
        changed_properties["color"] = (color1, color2)
        color_change = f"color change from {color1} to {color2}. "
        if color3 == color1:
            color3 = color2
    label = number3 + "_" + subject3 + "_" + position3 + "_" + color3 + "_" + object3
    if 'right' in label:
        label = label.replace('right', 'right_of')
    if 'left' in label:
        label = label.replace('left', 'left_of')
    return label


if __name__ == "__main__":
    data = json.load(open(f"{PROJ_ROOT}/data_path_all/single/union/hard_500.json", "r"))
    for i, item in enumerate(data):
        # remove the 3th and 4th image in the context_image list
        # del item["context_image"][2:4]
        # extract the transformation rule from the 1st and 2nd image
        if i==4:
            a = 1
        label = extract_transformation(item["context_image"][0], item["context_image"][1], item["context_image"][2])
        item["label"] = label
    with open(f"{PROJ_ROOT}/data_path_all/single/union/hard_500.json", "w") as f:
        json.dump(data, f, indent=4)