"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""
import os

import torch
from PIL import Image, ImageFile

import frcnn.utils as utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)
import json
import frcnn.transforms as T
from frcnn.engine import evaluate
from constants import category_to_label
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS

# Sacred init
SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Test Detection")
ex.observers.append(FileStorageObserver("sacred_test_detection"))


def load_saved_model(
    checkpoint_fpath: str,
    useWeights: bool,
    useChineseWeights: bool = False,
    device=torch.device("cpu"),
):
    num_classes = 25

    weights = None
    if useWeights:
        if useChineseWeights:
            print("using ChineseDataWeights")
            FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url = "https://download.pytorch.org/models/classification_model_state_dict.pth"
        else:
            print("using models.ResNet18_Weights.IMAGENET1K_V1")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8, weight_decay=0.0004)
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()

    return model


def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cocoJsonPath: str, transforms=None):

        self.transforms = transforms

        jFile = open(cocoJsonPath)
        self.data = json.load(jFile)
        jFile.close()

        print(f"Loading dataset with {len(self.data['images'])} images")

        ids = []
        for i, _ in enumerate(self.data["images"]):
            ids.append(i)

        self.imgs = ids

    def __getitem__(self, idx):
        # load images and masks
        image = self.data["images"][self.imgs[idx]]
        img_url = image["img_url"].split("/")
        image_file = img_url[-1]
        image_folder = img_url[-2]
        image_id = image["bln_id"]
        annotations = self.data["annotations"]
        boxes = []
        labels = []
        for annotation in annotations:
            if image_id == annotation["image_id"]:
                try:
                    labels.append(category_to_label[int(annotation["category_id"])])
                except:
                    continue
                x, y, w, h = annotation["bbox"]
                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h
                boxes.append([xmin, ymin, xmax, ymax])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = labels.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        src_folder = os.path.join("data", "testing", "images", "homer2")
        fname = os.path.join(src_folder, image_folder, image_file)
        img = Image.open(fname).convert("RGB")
        img.resize(
            (1000, round(img.size[1] * 1000.0 / float(img.size[0]))),
            Image.Resampling.BILINEAR,
        )
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


@ex.config
def my_config():
    checkpoint = ""
    useWeights = True
    cocoJsonPath = "./data/training/coco.json"


@ex.automain
def main(checkpoint: str, useWeights: bool, cocoJsonPath: str):
    if not checkpoint:
        print(
            "Check point not given. use `with checkpoint='<path>'` to provide the used checkpoint"
        )
        exit(1)

    # validate checkpoint and generate logfile
    if not os.path.exists(checkpoint):
        print("Check point path does not exists.")
        exit(1)

    if not os.path.isfile(checkpoint):
        print("Check point path is not a file")
        exit(1)

    if not checkpoint.endswith(".pt"):
        print("Check point path is not a model check point file")
        exit(1)

    if not "epoch_" in checkpoint:
        print("Check point path does not contain the epoch number")
        exit(1)

    print(f"Given checkpoint {checkpoint}")
    if not useWeights:
        print(
            "Using the model without the weights FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
        )

    print(f"Using coco file in {cocoJsonPath}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_saved_model(checkpoint, useWeights, device)
    dataset_test = Dataset(cocoJsonPath, transforms=get_transform())
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=utils.collate_fn,
    )

    evaluate(model, data_loader_test, device=device)
