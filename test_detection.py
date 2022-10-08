"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""
import os

import torch
from PIL import Image, ImageFile, ImageDraw, ImageFont

from cv2_utils import read_image_cv2, resize_image_cv2, save_image_cv2, show_image_cv2
from fs_utils import delete_dir_content

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms import transforms as T
from argparse import ArgumentParser
from pathlib import Path
from constants import label_to_char
from constants import model_input_size
import numpy as np


model_name = "model_detection.bak.pt"


def load_saved_model(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath , map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.FixedSizeCrop((672, 672)))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)


def get_image_from_bbox(
    image: cv2.Mat, bbox: torch.Tensor, name: str, persist=False
) -> cv2.Mat:
    xmin, ymin, xmax, ymax = bbox
    letter_image = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]
    letter_image = resize_image_cv2(letter_image, (model_input_size, model_input_size))

    # save if required
    if persist:
        fname = os.path.join("data", "cropped", f"{name}.png")
        save_image_cv2(letter_image, fname)

    return letter_image


def draw_boxes(
    img_path: str,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    img_size,
):
    cv2_image = read_image_cv2(img_path)
    cv2_image = resize_image_cv2(cv2_image, img_size)

    count = 0
    for box_t, score, label_t in zip(boxes, scores, labels):
        count += 1
        predicted_letter_from_detection = label_to_char[label_t.item()]
        print(
            {
                "letter": predicted_letter_from_detection,
                "prediction_score": score.item(),
            }
        )

        # draw a bot in the input image to visualize it
        xmin, ymin, xmax, ymax = box_t
        start = (int(xmin.item()), int(ymin.item()))
        end = (int(xmax.item()), int(ymax.item()))
        cv2_image = cv2.rectangle(cv2_image, start, end, (255, 0, 0), 1)

        cv2_image = put_asci_text(
            predicted_letter_from_detection, cv2_image, (start[0] + 10, start[1] - 10)
        )

    return cv2_image


def put_asci_text(letter: str, cv2_image, position):
    """
    adds the letter and the prediction score at the top of the bounding box
    """
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)

    # Draw non-ascii text onto image
    font = ImageFont.truetype("./assets/dejavu-fonts-ttf-2.37/ttf/DejaVuSerif.ttf", 10)
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, letter, font=font)

    # Convert back to Numpy array and switch back from RGB to BGR
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def main():

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="the image file")
    args = parser.parse_args()

    num_classes = 25
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8, weight_decay=0.0004)

    model, optimizer, start_epoch = load_saved_model(model_name, model, optimizer)
    model.eval()

    path = args.file

    size = (900, 900)
    image = Image.open(path).convert("RGB").resize(size)
    t = get_transform(False)
    image = t(image)

    result = model([image])
    boxes: torch.Tensor = result[0]["boxes"]
    labels: torch.Tensor = result[0]["labels"]
    scores: torch.Tensor = result[0]["scores"]

    cv2_image = draw_boxes(path, boxes, scores, labels, size)
    show_image_cv2(cv2_image)
    cv2.waitKey()
    cv2.waitKeyEx()


if __name__ == "__main__":
    main()
