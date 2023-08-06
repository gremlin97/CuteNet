import os
import json

def check_image_existence(file_path):
    return os.path.exists(file_path)

# Load the JSON file containing image information
with open("/scratch/kkasodek/controlNet/ControlNet/prompt-dirty.json", "r") as json_file:
    data = json.load(json_file)

# Assuming the image files are in the same directory as this script
i=0
for item in data:
    i+=1
    exists = check_image_existence('/scratch/kkasodek/controlNet/ControlNet/content/Dogs/source/'+item["source"])
    if exists == 'False':
        print(f'Image "{item["target"]}" exists: {exists}')

print(i)
