import os
import json


def get_num_test_images():
    coco = open(os.path.join("data", "testing", "coco.json"))
    data = json.load(coco)
    return len(data["images"])
