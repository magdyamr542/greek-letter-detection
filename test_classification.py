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

import cv2
import torch
from PIL import Image, ImageFile
from torchvision.datasets.dtd import PIL
from torchvision.transforms import transforms

from constants import categories, get_data_by_category, model_input_size
from cv2_utils import read_image_cv2, show_image_cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
num_classes = 24
model = None
model_path = "./data/training/classification_model.pt"


def crops_folder_exists(test_data_dir: str):
    assert os.path.exists(
        os.path.join(test_data_dir, "images")
    ), "Images directory does not exist!"
    if os.path.exists(os.path.join(test_data_dir, "crops")):
        return True
    else:
        return False


def create_data_for_testing(test_data_dir: str):
    os.chdir(test_data_dir)
    os.mkdir(os.path.join("crops"))
    os.mkdir(os.path.join("crops", "test"))
    try:
        f = open(glob.glob("*.json")[0])
    except:
        print("No json file was found!")
        raise

    data = json.load(f)
    f.close()
    l = []

    print(f"creating crops for {len(data['images'])} images")

    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        fname = os.path.join(img_url)
        image_id = image["bln_id"]
        try:
            Image.open(fname).convert("RGB")
            l.append(image_id)
        except:
            print(f"could not open image with id {image_id}")
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
                    (model_input_size, model_input_size), Image.BILINEAR
                )
                crop1.save(path, "JPEG", quality=85)


def load_model(model_path: str):
    theModel = torch.load(model_path, map_location=torch.device("cpu"))
    theModel.eval()
    return theModel


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


def classify(img_tensor: torch.Tensor):
    """
    classify gets an image tensor and returns the model classification as a tuple of (char as string , model prediction)
    """
    input_batch = img_tensor.unsqueeze(0)
    global model
    if not model:
        model = load_model(model_path)

    result: torch.Tensor = model(input_batch)
    max_index = result.argmax().item()
    _, __, char = get_data_by_category(categories[max_index])
    return char, result.max().item()


def main():

    data_dir = "data/testing"
    print("check if crop folder exists...")
    if not crops_folder_exists(data_dir):
        print("crop folder does not exist. make crops...")
        create_data_for_testing(data_dir)
        print("done making crops")


#     image = get_image(image_path)
#     image_tensor = get_pil_image_as_tensor(image)
#     char, pred = classify(image_tensor)
#     print(
#         {
#             "char": char,
#             "pred": pred,
#         }
#     )
#     cv2_image = read_image_cv2(image_path)
#     show_image_cv2(cv2_image, "char")
#     cv2.waitKey()


if __name__ == "__main__":
    main()
