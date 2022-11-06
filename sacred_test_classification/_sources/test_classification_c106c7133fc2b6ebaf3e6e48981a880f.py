"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""

import glob
import json
import os
import pickle
from multiprocessing.dummy import Pool

import cv2
import torch
from PIL import Image, ImageFile
from typing import List, Tuple, Any
import torch.nn as nn
from torchvision.datasets.dtd import PIL
from torchvision.transforms import transforms
from torchvision import models
from pathlib import Path
from constants import categories, get_data_by_category, model_input_size
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS

# Sacred init
SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Test Classification")
ex.observers.append(FileStorageObserver("sacred_test_classification"))


class CategoryEvaluationResult:
    def __init__(
        self,
        char: str,
        total_num_classifications: int,
        correct_num_classifications: int,
    ) -> None:
        self.char = char
        self.total_num_classifications = total_num_classifications
        self.correct_num_classifications = correct_num_classifications


ImageFile.LOAD_TRUNCATED_IMAGES = True


def crops_folder_exists(test_data_dir: str):
    assert os.path.exists(
        os.path.join(test_data_dir, "images")
    ), "Images directory does not exist!"
    if os.path.exists(os.path.join(test_data_dir, "crops")):
        return True
    else:
        return False


def create_data_for_testing(test_data_dir: str):
    cwd = os.getcwd()

    os.chdir(test_data_dir)
    os.mkdir(os.path.join("crops"))
    os.mkdir(os.path.join("crops", "test"))

    # get the coco.json file
    try:
        f = open(glob.glob("*.json")[0])
    except:
        print("No json file was found!")
        os.chdir(cwd)
        raise

    data = json.load(f)
    f.close()
    l = []

    print(f"Creating crops for {len(data['images'])} images...")

    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        fname = os.path.join(img_url)
        image_id = image["bln_id"]
        try:
            Image.open(fname).convert("RGB")
            l.append(image_id)
        except:
            print(f"Could not open image with id {image_id}")
            continue

    with open(os.path.join("image_list.bin"), "wb") as fp:  # Pickling
        pickle.dump(l, fp)

    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        image_id = image["bln_id"]
        fname = os.path.join(img_url)
        try:
            im = Image.open(fname).convert("RGB")
        except:
            print(f"Image {fname} does not exist")
            continue

        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                crop_id = annotation["id"]
                crop_filename = str(image_id) + "_" + str(crop_id) + ".jpg"
                x, y, w, h = annotation["bbox"]

                crop_directory = annotation["category_id"]
                crop_directory = os.path.join("crops", "test", str(crop_directory))
                if not os.path.exists(crop_directory):
                    os.mkdir(crop_directory)
                path = os.path.join(crop_directory, crop_filename)
                crop1 = im.crop((x, y, x + w, y + h))
                crop1 = crop1.resize(
                    (model_input_size, model_input_size), Image.Resampling.BILINEAR
                )
                crop1.save(path, "JPEG", quality=85)

    os.chdir(cwd)


def load_model(model_path: str):
    num_classes = 25
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_image(path: str) -> PIL.Image:
    # transform the img
    image = Image.open(path).convert("RGB")
    return image


def get_pil_image_as_tensor(image: PIL.Image) -> torch.Tensor:
    preprocess = transforms.Compose(
        [
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
        ]
    )
    return preprocess(image)


def cv2_image_to_pil_image(cv2_image: cv2.Mat) -> PIL.Image:
    return Image.fromarray(cv2_image)


def do_classify(image_path: str, model) -> str:
    """
    classify gets an image path and a model. it returns the classified label
    """

    image = get_image(image_path)
    image_tensor = get_pil_image_as_tensor(image)
    input_batch = image_tensor.unsqueeze(0)

    result: torch.Tensor = model(input_batch)
    max_index = result.argmax().item()
    _, __, char = get_data_by_category(categories[max_index])
    return char


def evaluate_model(model) -> int:
    """
    Returns:
        int: the model accuracy
    """
    categories_dir_path = os.path.join("data", "testing", "crops", "test")
    categories_dirs = os.listdir(categories_dir_path)

    all_crops = len(
        glob.glob(os.path.join(categories_dir_path, "**/*.jpg"), recursive=True)
    )

    print(f"Will classify {all_crops} crops")
    inputs = [
        (model, categories_dir_path, category_dir) for category_dir in categories_dirs
    ]

    p = Pool(len(inputs))

    results: List[CategoryEvaluationResult] = p.map(get_evaluation_for_category, inputs)

    return summarize_evaluation_results(results)


def summarize_evaluation_results(results: List[CategoryEvaluationResult]) -> int:
    """_summary_
    Returns:
        int: the model accuracy
    """
    total_classifications = 0
    total_correct_classifications = 0
    for result in results:
        total_classifications += result.total_num_classifications
        total_correct_classifications += result.correct_num_classifications
        char_accuracy = (
            result.correct_num_classifications / result.total_num_classifications
        )
        print(f"Accuracy for char {result.char} = {char_accuracy}")

    model_accuracy = total_correct_classifications / total_classifications
    print(f"Accuracy for model = {char_accuracy}")
    return model_accuracy


def get_evaluation_for_category(
    input: Tuple[Any, str, str]
) -> CategoryEvaluationResult:
    model, category_dir_base, category_dir = input
    crops = glob.glob(
        os.path.join(category_dir_base, category_dir, "**/*.jpg"), recursive=True
    )

    dir_total_num_classifications = 0
    dir_correct_num_classifications = 0

    _, __, char = get_data_by_category(int(category_dir))

    print(f"Classifying Category={category_dir} Crops={len(crops)} Char={char} ...")

    for crop in crops:

        model_char = do_classify(crop, model)

        if model_char == char:
            dir_correct_num_classifications += 1

        dir_total_num_classifications += 1

    print(f"Done with Category={category_dir} Crops={len(crops)} Char={char} ...")

    return CategoryEvaluationResult(
        char, dir_total_num_classifications, dir_correct_num_classifications
    )


@ex.config
def my_config():
    checkpoint = ""


@ex.automain
def main(checkpoint: str):
    Path(os.path.join("logs", "classification", "testing")).mkdir(
        parents=True, exist_ok=True
    )

    if not checkpoint:
        print(
            "check point not given. use `with checkpoint='<path>'` to provide the used checkpoint"
        )
        exit(1)

    # validate checkpoint and generate logfile
    if not os.path.exists(checkpoint):
        print("check point path does not exists.")
        exit(1)

    if not os.path.isfile(checkpoint):
        print("check point path is not a file")
        exit(1)

    if not checkpoint.endswith(".pt"):
        print("check point path is not a model check point file")
        exit(1)

    if not "epoch_" in checkpoint:
        print("check point path does not contain the epoch number")
        exit(1)

    epoch_num = checkpoint[
        checkpoint.find("epoch_") + len("epoch_") : checkpoint.find(".pt")
    ]
    try:
        epoch_num = int(epoch_num)
    except:
        print("could not parse the epoch number from the check point")

    # prepare data for testing
    data_dir = "data/testing"
    print("Check if crop folder exists...")
    if not crops_folder_exists(data_dir):
        print("Crop folder does not exist. make crops...")
        create_data_for_testing(data_dir)
        print("Done making crops")

    # evaluate the model
    print("Start evaluating classifier...")
    model = load_model(checkpoint)
    evaluate_model(model)
