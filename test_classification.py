"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""

import glob
import os
from multiprocessing.dummy import Pool

from torchvision import datasets, models, transforms
import cv2
import torch
from PIL import Image, ImageFile
from typing import Dict, List, Tuple, Any
import torch.nn as nn
from torchvision.datasets.dtd import PIL
from torchvision.transforms import transforms
from torchvision import models
from pathlib import Path
from constants import class_to_index_chinese_data, model_input_size
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
    if os.path.exists(os.path.join(test_data_dir, "crops", "test")):
        return True
    else:
        return False


def create_data_for_testing(test_data_dir: str):
    os.mkdir(os.path.join(test_data_dir, "crops", "test"))

    testImages = glob.glob(os.path.join(test_data_dir, "images", "testing", "**/*.png"))
    print("test images", len(testImages))
    imgLeft = len(testImages)
    for img in testImages:
        imgDir = img.split("/")[-2]
        imgName = img.split("/")[-1]
        if not os.path.exists(os.path.join(test_data_dir, "crops", "test", imgDir)):
            os.makedirs(os.path.join(test_data_dir, "crops", "test", imgDir))

        try:
            im = Image.open(img).convert("RGB")
            im = im.resize((model_input_size, model_input_size))
            im.save(os.path.join(test_data_dir, "crops", "test", imgDir, imgName))
            imgLeft -= 1
            if imgLeft % 10 == 0:
                print("imgs left", imgLeft)
        except:
            print("error while saving image", img)


@ex.capture
def load_model(data_dir: str, model_path: str):
    num_classes = len(os.listdir(os.path.join(data_dir, "images", "training")))
    print("num classes", num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = models.resnet18()
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


def do_classify(image_path: str, model, category_index_to_category={}) -> str:
    """
    classify gets an image path and a model. it returns the classified label
    """

    image = get_image(image_path)
    image_tensor = get_pil_image_as_tensor(image)
    input_batch = image_tensor.unsqueeze(0)

    result: torch.Tensor = model(input_batch)
    max_index = result.argmax().item()
    return category_index_to_category[max_index]


def evaluate_model(model) -> int:
    """
    Returns:
        int: the model accuracy
    """
    categories_dir_path = os.path.join("chinese-data", "crops", "test")
    categories_dirs = os.listdir(categories_dir_path)

    all_crops = len(
        glob.glob(os.path.join(categories_dir_path, "**/*.png"), recursive=True)
    )

    class_to_index = datasets.ImageFolder(
        os.path.join("chinese-data", "crops", "train")
    ).class_to_idx

    print(f"Will classify {all_crops} crops")
    inputs = [
        (model, categories_dir_path, category_dir, class_to_index)
        for category_dir in categories_dirs
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
    input: Tuple[Any, str, str, Dict[str, int]]
) -> CategoryEvaluationResult:
    model, category_dir_base, category_dir, class_to_index_map = input
    crops = glob.glob(
        os.path.join(category_dir_base, category_dir, "**/*.png"), recursive=True
    )

    dir_total_num_classifications = 0
    dir_correct_num_classifications = 0

    char = category_dir

    index_to_class = {v: k for k, v in class_to_index_map.items()}

    print(f"Classifying Category={category_dir} Crops={len(crops)}")

    for crop in crops:

        model_char = do_classify(crop, model, index_to_class)

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
    useWeights = True
    all_categories = os.listdir(os.path.join("chinese-data", "images", "training"))


@ex.automain
def main(checkpoint: str):
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
    data_dir = "chinese-data"
    print("Check if crop folder exists...")
    if not crops_folder_exists(data_dir):
        print("Crop folder does not exist. make crops...")
        create_data_for_testing(data_dir)
        print("Done making crops")

    # evaluate the model
    print("Start evaluating classifier...")
    model = load_model(data_dir, checkpoint)
    evaluate_model(model)
