"""
Generate union, intersection, and difference task JSON files for the mix task.

Samples and organizes image triples from the CARV dataset into task definition
JSONs stored under data_path_all/mix/{operation}/. These JSONs are consumed by
the inference script during evaluation.
"""
PROJ_ROOT = "/data/yongka/analogy/spatial"
DATA_PATH = f"{PROJ_ROOT}/data_path_all"
import json
import os
import random
import copy

COLOR = ['red', 'blue', 'wood']
NUMBER = ['one', 'two']
POSITION = ["left_of", "right_of", "on", "under"]
OBJECT = ['chair', 'table']
DATA_TAMPLATE = {
    "number": "",
    "subject": "",
    "position": "",
    "color": "",
    "object": "",
}

def extract_object():
    # Example: directory containing your images
    image_dir = f"{PROJ_ROOT}/data/subject"

    chair_objects = set()
    armchair_objects = set()
    table_objects = set()

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpeg") and '_' in filename:
            # Remove extension
            name = filename.replace(".jpeg", "")
            # Split by underscore
            parts = name.split("_")
            # First part is always the main object (even if it has hyphens)
            head = parts[0]
            tail = parts[-1]
            
            if tail == "chair":
                chair_objects.add(head)
            elif tail == "armchair":
                armchair_objects.add(head)
            elif tail == "table":
                table_objects.add(head)

    # Convert to sorted list for readability
    chair_objects = sorted(chair_objects)
    armchair_objects = sorted(armchair_objects)
    table_objects = sorted(table_objects)
    return chair_objects, armchair_objects, table_objects

def shuffle_options(options):
    wrong_position = options[0]
    wrong_subject = options[1]
    ground_truth = options[2]
    random.shuffle(options)
    wrong_position_idx = options.index(wrong_position)
    wrong_subject_idx = options.index(wrong_subject)
    answer_idx = options.index(ground_truth) # start from 0
    return options, wrong_position_idx, wrong_subject_idx, answer_idx

def get_random_datapoint():
    dp = copy.deepcopy(DATA_TAMPLATE)
    dp['number'] = random.choice(NUMBER)
    dp['subject'] = random.choice(SUBJECT)
    dp['position'] = random.choice(POSITION)
    dp['object'] = random.choice(OBJECT)
    dp['color'] = random.choice(COLOR)
    return dp

def get_transformed_datapoint(original_dp, rule):
    transformed_dp = copy.deepcopy(original_dp)
    for trans, v in rule.items():
        transformed_dp[trans] = v
    return transformed_dp

def get_hard_datapoint(original_dp, shared_trans, distinct_trans1, distinct_trans2):
    a1_h = copy.deepcopy(original_dp)
    a2_h = copy.deepcopy(original_dp)
    property_pool = list(DATA_TAMPLATE.keys())
    for trans in shared_trans+distinct_trans1+distinct_trans2:
        property_pool.remove(trans)
    # change one more property in property_pool for hard case
    extra_properties = random.sample(property_pool, 2)
    v_extra_1 = random.choice([value for value in VALUE_DICT[extra_properties[0]] if value != original_dp[extra_properties[0]]])
    a1_h[extra_properties[0]] = v_extra_1
    v_extra_2 = random.choice([value for value in VALUE_DICT[extra_properties[1]] if value != original_dp[extra_properties[1]]])
    a2_h[extra_properties[1]] = v_extra_2
    return a1_h, a2_h

def get_transformation_rules(original_dp, shared_trans, distinct_trans1, distinct_trans2):
    rule1 = {}
    rule2 = {}
    label_union = copy.deepcopy(original_dp)
    label_intersection = copy.deepcopy(original_dp)
    for trans in shared_trans:
        v = random.choice([value for value in VALUE_DICT[trans] if value != original_dp[trans]])
        rule1[trans] = v
        rule2[trans] = v
        label_union[trans] = v
        label_intersection[trans] = v
    for trans in distinct_trans1:
        v1 = random.choice([value for value in VALUE_DICT[trans] if value != original_dp[trans]])
        rule1[trans] = v1
        label_union[trans] = v1
    for trans in distinct_trans2:
        v2 = random.choice([value for value in VALUE_DICT[trans] if value != original_dp[trans]])
        rule2[trans] = v2
        label_union[trans] = v2
    return rule1, rule2, label_union, label_intersection

def dict_to_filename(dp):
    filename = f"{dp['number']}_{dp['subject']}_{dp['position']}_{dp['color']}_{dp['object']}"
    return filename

def generate_new_data(num_samples, num_shared, num_distinct, context_images=[]):
    u_easy_data = []
    u_hard_data = []
    i_easy_data = []
    i_hard_data = []
    # context_images = []
    generate_hard = False
    if num_distinct == 1 and num_shared == 1:
        generate_hard = True
    while len(u_easy_data) < num_samples:
        # choose 1 shared property
        shared_trans = random.sample(list(DATA_TAMPLATE.keys()), num_shared)
        rest_properties = [prop for prop in DATA_TAMPLATE.keys() if prop not in shared_trans]
        distinct_trans_1 = random.sample(rest_properties, num_distinct)
        distinct_trans_2 = random.sample([prop for prop in rest_properties if prop not in distinct_trans_1], num_distinct)
        
        # randomly initialize anchor image A1
        a1_e = get_random_datapoint()
        rule1, rule2, label_union, label_intersection = get_transformation_rules(a1_e, shared_trans, distinct_trans_1, distinct_trans_2)
        b1_e = get_transformed_datapoint(a1_e, rule1)
        b2_e = get_transformed_datapoint(a1_e, rule2)
        if [a1_e, b1_e, a1_e, b2_e] in context_images:
            continue
        context_images.append([a1_e, b1_e, a1_e, b2_e])
        u_easy_data.append({"context_image": [dict_to_filename(a1_e), dict_to_filename(b1_e), dict_to_filename(a1_e), dict_to_filename(b2_e), dict_to_filename(a1_e)], "label": dict_to_filename(label_union)})
        i_easy_data.append({"context_image": [dict_to_filename(a1_e), dict_to_filename(b1_e), dict_to_filename(a1_e), dict_to_filename(b2_e), dict_to_filename(a1_e)], "label": dict_to_filename(label_intersection)})
        
        if generate_hard:
            a1_h, a2_h = get_hard_datapoint(a1_e, shared_trans, distinct_trans_1, distinct_trans_2)
            b1_h = get_transformed_datapoint(a1_h, rule1)
            b2_h = get_transformed_datapoint(a2_h, rule2)
        
            if [a1_h, b1_h, a2_h, b2_h] in context_images:
                continue
            context_images.append([a1_h, b1_h, a2_h, b2_h])
        
            u_hard_data.append({"context_image": [dict_to_filename(a1_h), dict_to_filename(b1_h), dict_to_filename(a2_h), dict_to_filename(b2_h), dict_to_filename(a1_e)], "label": dict_to_filename(label_union)})
            i_hard_data.append({"context_image": [dict_to_filename(a1_h), dict_to_filename(b1_h), dict_to_filename(a2_h), dict_to_filename(b2_h), dict_to_filename(a1_e)], "label": dict_to_filename(label_intersection)})
        
        
        # print(rule1)
        # print(rule2)
        # print(dict_to_filename(a1_e), dict_to_filename(b1_e), dict_to_filename(a1_e), dict_to_filename(b2_e))
        # if generate_hard:
        #     print(dict_to_filename(a1_h), dict_to_filename(b1_h), dict_to_filename(a2_h), dict_to_filename(b2_h))
        # print(dict_to_filename(label_union), dict_to_filename(label_intersection))
        a = 1

    save_data(u_easy_data, "mix", "union", "easy", num_samples)
    save_data(i_easy_data, "mix", "intersection", "easy", num_samples)
    if generate_hard:
        save_data(u_hard_data, "mix", "union", "hard", num_samples)
        save_data(i_hard_data, "mix", "intersection", "hard", num_samples)
    return

def generate_new_data_diff(num_samples, num_shared, num_distinct, context_images=[]):
    d_easy_data = []
    d_hard_data = []
    # i_easy_data = []
    # i_hard_data = []
    # context_images = []
    generate_hard = True
    # if num_distinct == 1 and num_shared == 1:
    #     generate_hard = True
    while len(d_easy_data) < num_samples:
        # choose 1 shared property
        shared_trans = random.sample(list(DATA_TAMPLATE.keys()), num_shared)
        rest_properties = [prop for prop in DATA_TAMPLATE.keys() if prop not in shared_trans]
        distinct_trans_1 = random.sample(rest_properties, num_distinct)
        distinct_trans_2 = random.sample([prop for prop in rest_properties if prop not in distinct_trans_1], num_distinct)
        
        # randomly initialize anchor image A1
        a1_e = get_random_datapoint()
        rule1, _, label_union, label_intersection = get_transformation_rules(a1_e, shared_trans, distinct_trans_1, distinct_trans_2)
        # split the rule1 to get rule2 by removing one transformation
        if len(rule1) ==2:
            len_rule2 = 1
        else:
            len_rule2 = 2
        rule2 = random.sample(list(rule1.items()), len_rule2)
        rule2 = dict(rule2)
        # get the removed transformation and update label_union accordingly
        removed_trans = dict([item for item in rule1.items() if item not in rule2.items()])
        label_diff = get_transformed_datapoint(a1_e, removed_trans)
        
        
        b1_e = get_transformed_datapoint(a1_e, rule1)
        b2_e = get_transformed_datapoint(a1_e, rule2)
        if [a1_e, b1_e, a1_e, b2_e] in context_images:
            continue
        context_images.append([a1_e, b1_e, a1_e, b2_e])
        d_easy_data.append({"context_image": [dict_to_filename(a1_e), dict_to_filename(b1_e), dict_to_filename(a1_e), dict_to_filename(b2_e), dict_to_filename(a1_e)], "label": dict_to_filename(label_diff)})
        # i_easy_data.append({"context_image": [dict_to_filename(a1_e), dict_to_filename(b1_e), dict_to_filename(a1_e), dict_to_filename(b2_e), dict_to_filename(a1_e)], "label": dict_to_filename(label_intersection)})
        
        if generate_hard:
            a1_h, a2_h = get_hard_datapoint(a1_e, shared_trans, distinct_trans_1, distinct_trans_2)
            b1_h = get_transformed_datapoint(a1_h, rule1)
            b2_h = get_transformed_datapoint(a2_h, rule2)
        
            if [a1_h, b1_h, a2_h, b2_h] in context_images:
                continue
            context_images.append([a1_h, b1_h, a2_h, b2_h])
        
            d_hard_data.append({"context_image": [dict_to_filename(a1_h), dict_to_filename(b1_h), dict_to_filename(a2_h), dict_to_filename(b2_h), dict_to_filename(a1_e)], "label": dict_to_filename(label_diff)})
            # i_hard_data.append({"context_image": [dict_to_filename(a1_h), dict_to_filename(b1_h), dict_to_filename(a2_h), dict_to_filename(b2_h), dict_to_filename(a1_e)], "label": dict_to_filename(label_intersection)})
        
        
        # print(rule1)
        # print(rule2)
        # print(dict_to_filename(a1_e), dict_to_filename(b1_e), dict_to_filename(a1_e), dict_to_filename(b2_e))
        # if generate_hard:
        #     print(dict_to_filename(a1_h), dict_to_filename(b1_h), dict_to_filename(a2_h), dict_to_filename(b2_h))
        # print(dict_to_filename(label_union), dict_to_filename(label_intersection))
        a = 1
    if num_shared == 1 and num_distinct ==1:
        save_data(d_easy_data, "mix", "difference", "easy", num_samples)
    # elif num_shared ==2 and num_distinct ==1:
    #     save_data(d_easy_data, "large", "difference", "easy", num_samples)
    # elif num_shared ==3 and num_distinct ==1:
    #     save_data(d_easy_data, "extra_large", "difference", "hard", num_samples)
    # save_data(i_easy_data, "mix", "intersection", "easy", num_samples)
    if generate_hard:
        save_data(d_hard_data, "mix", "difference", "hard", num_samples)
    #     save_data(i_hard_data, "mix", "intersection", "hard", num_samples)
    return

def save_data(data, task_type, operation, difficulty, num_samples):
    # check if data path exists
    # task_type = "large"
    # difficulty = "n"
    folder_path = f"{DATA_PATH}/{task_type}/{operation}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = f"{folder_path}/{difficulty}_{num_samples}.json"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    else:
        # read existing data
        with open(file_path, "r") as f:
            existing_data = json.load(f)
        # append new data
        existing_data.extend(data)
        # save back
        file_path = f"{folder_path}/{difficulty}_3_{len(existing_data)}.json"
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)
    print(f"Saved {len(data)} data to json file.")
    return

if __name__ == "__main__":
    chair_objects, armchair_objects, table_objects = extract_object()
    table_objects.remove("chair")  # remove chair from table objects
    table_objects.remove("chair2")  # remove chair from table objects
    intersection_objects = sorted(set(chair_objects) & set(table_objects))
    intersection_objects.remove("painting")
    intersection_objects.remove("ball-of-yarn")
    intersection_objects.remove("plate")
    intersection_objects.remove("scarf")
    intersection_objects.remove("dragonfruit")
    intersection_objects.remove("lemon")
    intersection_objects.remove("pillow")
    intersection_objects.remove("plant")
    # intersection_objects.remove("toilet-roll")
    intersection_objects.remove("helmet")
    intersection_objects.remove("book")
    intersection_objects.remove("cup")
    intersection_objects.remove("lamp")
    intersection_objects.remove("mug")
    intersection_objects.remove("wineglass")
    intersection_objects.remove("kettle")
    intersection_objects.remove("sunglasses")
    # print(intersection_objects)
    SUBJECT = intersection_objects
    VALUE_DICT = {
        "subject": SUBJECT,
        "position": POSITION,
        "object": OBJECT,
        "color": COLOR,
        "number": NUMBER
    }
    # file_path_easy = f"{DATA_PATH}/large/union/n_3.json"
    # data_easy = json.load(open(file_path_easy))
    # file_path_hard = f"{DATA_PATH}/mix/union/hard_200.json"
    # data_hard = json.load(open(file_path_hard))
    # context_images = [item['context_image'][:-1] for item in data_easy]
    
    for num_samples in [500]:
        generate_new_data(num_samples, 1, 1)
        # generate_new_data_diff(num_samples, 2, 1)
        # generate_new_data_diff(num_samples, 3, 1)
