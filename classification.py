"""
This code base on the official Pytorch Transfer Learning for Computer Vision Tutorial
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

A Resnet18 was trained during 30 epochs during 44m 43s.
The final train and validation loss and accuracy were:
Train: Loss 0.4192 and Accuracy 0.87
Validation: Loss 0.6916 and Accuracy 0.81
"""


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
from PIL import Image
import json
import os, glob, pickle
from sklearn.model_selection import train_test_split
from PIL import ImageFile
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
import shutil

from utils.get_num_test_images import get_num_test_images

# Sacred init
SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Train Classification")
ex.observers.append(FileStorageObserver("sacred_train_classification"))

ImageFile.LOAD_TRUNCATED_IMAGES = True
from constants import model_input_size

check_point_name = "classification_model_check_point.pt"


def update_model_with_saved_checkpoint(checkpoint_fpath, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint["epoch"]
    except Exception as e:
        return None


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    check_point_save_path: str,
    device=None,
    num_epochs=25,
    start_epoch=-1,
):
    since = time.time()

    print(f"Start training at {since}")

    for epoch in range(start_epoch + 1, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                print("Training...")
            else:
                model.eval()  # Set model to evaluate mode
                print("Evaluating...")

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # stats per epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == "val" and scheduler is not None:
                scheduler.step()
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

        print(f"Saving checkpoint for epoch {epoch}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            check_point_save_path,
        )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model


@ex.capture
def initialize_model(num_classes, feature_extract=False):
    """
    - initializes the resnet18 model replacing the last fully connected layer (fc) with a linear layer that maps to the 24 chars we have.
    """

    model = models.resnet18()
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False  # it's pretrained.

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(
        num_ftrs, num_classes
    )  # only this layer will be trained with our data.
    return model


def crops_folder_exist(data_dir):
    assert os.path.exists(
        os.path.join(data_dir, "images")
    ), "Data directory does not exist!"
    if os.path.exists(os.path.join(data_dir, "crops")):
        return True
    else:
        return False


def create_data(data_dir: str):
    os.mkdir(os.path.join(data_dir, "crops"))
    os.mkdir(os.path.join(data_dir, "crops", "train"))
    os.mkdir(os.path.join(data_dir, "crops", "val"))

    allTrainImages = glob.glob(os.path.join(data_dir, "images", "training", "**/*.png"))
    print("all images", len(allTrainImages))
    train, val = train_test_split(allTrainImages, random_state=4)
    print("training images", len(train))
    print("validation images", len(val))
    valLeft = len(val)
    for valImg in val:
        imgDir = valImg.split("/")[-2]
        imgName = valImg.split("/")[-1]
        if not os.path.exists(os.path.join(data_dir, "crops", "val", imgDir)):
            os.makedirs(os.path.join(data_dir, "crops", "val", imgDir))

        try:
            im = Image.open(valImg).convert("RGB")
            im = im.resize((model_input_size, model_input_size))
            im.save(os.path.join(data_dir, "crops", "val", imgDir, imgName))
            valLeft -= 1
            print("val left", valLeft)
        except:
            print("error while saving image", valImg)

    trainLeft = len(val)
    for trainImg in train:
        imgDir = trainImg.split("/")[-2]
        imgName = trainImg.split("/")[-1]
        if not os.path.exists(os.path.join(data_dir, "crops", "train", imgDir)):
            os.makedirs(os.path.join(data_dir, "crops", "train", imgDir))

        try:
            im = Image.open(trainImg).convert("RGB")
            im = im.resize((model_input_size, model_input_size))
            im.save(os.path.join(data_dir, "crops", "val", imgDir, imgName))
            trainLeft -= 1
            print("train left", trainLeft)
        except:
            print("error while saving image", trainImg)


@ex.config
def my_config():
    checkpoint = ""
    epochs = None
    useWeights = True
    trainedModelName = (
        "classification_model_checkpoint_{noWeightPrefix}epoch_{numEpochs}.pt".format(
            noWeightPrefix="no_weights_" if not useWeights else "", numEpochs=epochs
        )
    )
    trainedModelSaveBasePath = os.path.join(
        "chinese-data",
        "saved_checkpoints",
    )
    trainedModelSavePath = os.path.join(
        trainedModelSaveBasePath,
        trainedModelName,
    )


@ex.automain
def main(
    checkpoint: str,
    epochs: int,
    trainedModelSavePath: str,
    trainedModelSaveBasePath: str,
):
    Path(trainedModelSaveBasePath).mkdir(parents=True, exist_ok=True)

    data_dir = "chinese-data"
    num_classes = len(os.listdir(os.path.join(data_dir, "images", "training")))
    print("number of classes is ", num_classes)
    batch_size = 40

    if not epochs:
        print(
            "epochs not given. use `with epochs='<number>'` to provide the num of epochs used while training"
        )
        exit(1)

    num_epochs = int(epochs)

    print(f"Num of epochs given {num_epochs}")
    if not checkpoint:
        print("No checkpoint given")
    else:
        print(f"Checkpoint given {checkpoint}")
    print(f"Will save the trained model to {trainedModelSavePath}")

    if not crops_folder_exist(data_dir):
        print(f"No crops found. will create them ...")
        create_data(data_dir)
        print(f"Crops were created")

    # validate checkpoint
    if checkpoint and not os.path.exists(checkpoint):
        print("Check point path does not exists.")
        exit(1)

    if checkpoint and not os.path.isfile(checkpoint):
        print("Check point path is not a file")
        exit(1)

    if checkpoint and not checkpoint.endswith(".pt"):
        print("Check point path is not a model check point file")
        exit(1)

    # Augmentations and data transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(
                    model_input_size, scale=(0.9, 1.1), ratio=(0.9, 1.1)
                ),
                transforms.RandomAdjustSharpness(2),
                transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [transforms.Resize(model_input_size), transforms.ToTensor()]
        ),
    }

    # Create training and validation datasets
    image_datasets = {
        "train": datasets.ImageFolder(
            os.path.join(data_dir, "crops", "train"), data_transforms["train"]
        ),
        "val": datasets.ImageFolder(
            os.path.join(data_dir, "crops", "val"), data_transforms["val"]
        ),
    }

    # Create training and validation dataloaders
    dataloaders_dict = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2
        ),  # TODO: change in gpu
        "val": torch.utils.data.DataLoader(
            image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2
        ),
    }

    # Initialize the model
    feature_extract = False  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
    model = initialize_model(num_classes, feature_extract)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {str(device)}")

    # Send the model to GPU
    model = model.to(device)

    # Initialize the model to optimize all parameters
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, weight_decay=0.0004, momentum=0.8
    )

    start_epoch = -1
    if checkpoint:
        result = update_model_with_saved_checkpoint(checkpoint, model, optimizer)

        if result:
            print("Found a saved model. will continue from the saved checkpoint")
            model, optimizer, saved_model_epochs = result
            print(f"Start epoch of saved checkpoint {saved_model_epochs}")
            if num_epochs < saved_model_epochs:
                print(
                    f"Saved start epoch {saved_model_epochs} is bigger than given number of epochs {num_epochs}"
                )
                exit(1)
            else:
                start_epoch = saved_model_epochs
    else:
        print("No saved checkpoint found. will start training from the beginning...")

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    # Train and evaluate
    model = train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        scheduler,
        device=device,
        check_point_save_path=trainedModelSavePath,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
    )
