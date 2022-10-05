"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""

from torchvision.datasets.dtd import PIL
from torchvision.transforms import transforms
from PIL import Image, ImageFile
import torch
from PIL import ImageFile
from argparse import ArgumentParser
from cv2_utils import read_image_cv2, show_image_cv2
import cv2
from constants import get_data_by_category, model_input_size
from constants import categories


ImageFile.LOAD_TRUNCATED_IMAGES = True
num_classes = 24
model = None
model_path = "clas_model.pt"


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
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="the image file")
    args = parser.parse_args()
    image_path = args.file

    image = get_image(image_path)
    image_tensor = get_pil_image_as_tensor(image)
    char, pred = classify(image_tensor)
    print(
        {
            "char": char,
            "pred": pred,
        }
    )
    cv2_image = read_image_cv2(image_path)
    show_image_cv2(cv2_image, "char")
    cv2.waitKey()


if __name__ == "__main__":
    main()
