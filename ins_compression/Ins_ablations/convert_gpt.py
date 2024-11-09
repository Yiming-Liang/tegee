import json 
import os
train_files = []
test_files = []

with open("/ML-A100/home/gezhang/tmp_xw/mapping.json", "r") as fp:
    data = json.load(fp)
# Iterate through the dictionary
for key, value in data.items():
    # Check if the category is "test"
    if value["category"] == "train":
        # Add the filename to the list
        train_files.append(value["filename"])
    if value["category"] == "test":
        test_files.append(value["filename"])


def loadfile(train_path):
    with open(os.path.join('/ML-A100/home/gezhang/default/', train_path), encoding='UTF-8') as f:
        task = json.load(f)
    ins = task['Instances']
    length = len(ins)
    df = pd.DataFrame(ins)
    if len(task['Definition']) > 1:
        merged = " ".join([string for string in task['Definition'] if string])
        task['Definition'] = [merged]
    return task['Definition']