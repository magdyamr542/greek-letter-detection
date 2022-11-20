"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""
import os
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import json
from PIL import ImageFile

from utils.get_num_test_images import get_num_test_images


ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import frcnn.transforms as T
from frcnn.engine import train_one_epoch, evaluate
import frcnn.utils as utils
from constants import category_to_label
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS

# Sacred init
SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Train Detection")
ex.observers.append(FileStorageObserver("sacred_train_detection"))


def update_model_with_saved_checkpoint(checkpoint_fpath, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint["epoch"]
    except:
        return None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None, isTrain=False):
        self.transforms = transforms

        jFile = open(os.path.join("data", "training", "coco.json"))
        self.data = json.load(jFile)
        jFile.close()

        img_ids = list(map(lambda item: item["id"], self.data["images"]))
        train, val = train_test_split(img_ids, random_state=8)
        if isTrain:
            imgs = list(train)
        else:
            imgs = list(val)

        ids = []
        for i, image in enumerate(self.data["images"]):
            if image["bln_id"] in imgs:
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

        src_folder = os.path.join("data", "training", "images", "homer2")
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


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.FixedSizeCrop((672, 672)))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)


@ex.config
def my_config():
    checkpoint = ""
    epochs = None
    useWeights = True
    useChineseWeights = False
    numTestImages = get_num_test_images()
    trainedModelName = (
        "detection_model_checkpoint_{noWeightPrefix}epoch_{numEpochs}.pt".format(
            noWeightPrefix="no_weights_" if not useWeights else "", numEpochs=epochs
        )
    )
    trainedModelSaveBasePath = os.path.join(
        "data",
        "training",
        "saved_detection_checkpoints",
        f"{numTestImages}_test_images",
    )

    trainedModelSavePath = os.path.join(
        trainedModelSaveBasePath,
        trainedModelName,
    )


@ex.automain
def main(
    checkpoint: str,
    epochs: Optional[int],
    numTestImages: int,
    useWeights: bool,
    useChineseWeights: bool,
    trainedModelSavePath: str,
    trainedModelSaveBasePath: str,
):

    Path(trainedModelSaveBasePath).mkdir(parents=True, exist_ok=True)

    if not epochs:
        print(
            "epochs not given. use `with epochs='<number>'` to provide the num of epochs used while training"
        )
        exit(1)

    if checkpoint:
        print(f"Checkpoint given {checkpoint}")

    print(f"Num of epochs given {epochs}")
    print(f"Num of test images {numTestImages}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device {str(device)}")

    num_classes = len(category_to_label) + 1
    dataset = Dataset(transforms=get_transform(True), isTrain=True)
    dataset_test = Dataset(transforms=get_transform(False), isTrain=False)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=utils.collate_fn,
    )

    if not useWeights:
        print("Using the model without the weights FasterRCNN_ResNet50_FPN_Weights")

    weights = None
    if useWeights:
        if useChineseWeights:
            print("using ChineseDataWeights")
            FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url = "https://download.pytorch.org/models/classification_model_state_dict_epoch_50.pth"
        else:
            print("using models.ResNet18_Weights.IMAGENET1K_V1")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8, weight_decay=0.0004)
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85)

    start_epoch = -1
    if checkpoint:
        result = update_model_with_saved_checkpoint(checkpoint, model, optimizer)

        if result:
            print("Found a saved model. will continue from the saved checkpoint")
            model, optimizer, saved_model_epochs = result
            print(f"Start epoch of saved checkpoint {saved_model_epochs}")
            if epochs < saved_model_epochs:
                print(
                    f"Saved start epoch {saved_model_epochs} is bigger than given number of epochs {num_epochs}"
                )
                exit(1)
            else:
                start_epoch = saved_model_epochs
    else:
        print("No saved checkpoint found. will start training from the beginning...")

    print(f"Will be saving the trained model to {trainedModelSavePath}")
    if os.path.exists(trainedModelSavePath):
        print(f"a model already exists at {trainedModelSavePath}")
        exit(1)

    for epoch in range(start_epoch + 1, epochs):
        train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=10,
        )
        if epoch < 4:
            lr_scheduler1.step()
        elif epoch > 40 and epoch < 48:
            lr_scheduler2.step()
        evaluate(model, data_loader_test, device=device)
        print(f"Saving check point for epoch {epoch}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            trainedModelSavePath,
        )

    print("That's it!")
