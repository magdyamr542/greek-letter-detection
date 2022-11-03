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
from argparse import ArgumentParser
from logging import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
from PIL import Image
import json
import os, glob, pickle
from sklearn.model_selection import train_test_split
from logger_utils import getLogger
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from constants import model_input_size

check_point_name = "classification_model_check_point.pt"


def update_model_with_saved_checkpoint(checkpoint_fpath, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_fpath, model, optimizer)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint["epoch"]
    except:
        return None


def train_model(
    logger: Logger,
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    start_epoch=-1,
):
    since = time.time()

    logger.info(f"Start training at {since}")

    for epoch in range(start_epoch + 1, num_epochs):
        logger.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logger.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                logger.info("Training...")
            else:
                model.eval()  # Set model to evaluate mode
                logger.info("Evaluating...")

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
            logger.info(
                "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )

        logger.info(f"Saving checkpoint for epoch {epoch}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            check_point_name,
        )

    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model


def initialize_model(num_classes, feature_extract=False):
    """
    - initializes the resnet18 model replacing the last fully connected layer (fc) with a linear layer that maps to the 24 chars we have.
    """
    model = models.resnet18(pretrained=True)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False  # it's pretrained.

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(
        num_ftrs, num_classes
    )  # only this layer will be trained with our data.
    return model


def crops_folder_exist(data_dir):
    os.chdir(os.path.join(data_dir))
    assert os.path.exists(os.path.join("images")), "Data directory does not exist!"
    if os.path.exists(os.path.join("crops")):
        return True
    else:
        return False


def create_data():
    os.mkdir(os.path.join("crops"))
    os.mkdir(os.path.join("crops", "train"))
    os.mkdir(os.path.join("crops", "val"))
    try:
        f = open(glob.glob("*.json")[0])
    except:
        logger.info("No json file was found!")
    data = json.load(f)
    f.close()
    l = []
    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        fname = os.path.join(img_url)
        image_id = image["bln_id"]
        try:
            Image.open(fname).convert("RGB")
            l.append(image_id)
        except:
            continue
    with open(os.path.join("image_list.bin"), "wb") as fp:  # Pickling
        pickle.dump(l, fp)

    train, val = train_test_split(l, random_state=8)

    for i, image in enumerate(data["images"]):
        img_url = image["img_url"][2:]
        image_id = image["bln_id"]
        fname = os.path.join(img_url)
        try:
            im = Image.open(fname).convert("RGB")
        except:
            logger.info(f"File not found {fname}")
            continue
        if image_id in val:
            split = "val"
        else:
            split = "train"
        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                crop_id = annotation["id"]
                crop_filename = str(image_id) + "_" + str(crop_id) + ".jpg"
                x, y, w, h = annotation["bbox"]

                crop_directory = annotation["category_id"]
                if crop_directory in [108, 208, 31, 76, 176]:
                    continue
                crop_directory = os.path.join("crops", split, str(crop_directory))
                if not os.path.exists(crop_directory):
                    os.mkdir(crop_directory)
                path = os.path.join(crop_directory, crop_filename)
                crop1 = im.crop((x, y, x + w, y + h))
                crop1 = crop1.resize(
                    (model_input_size, model_input_size), Image.BILINEAR
                )
                crop1.save(path, "JPEG", quality=85)


if __name__ == "__main__":
    os.chdir(os.getcwd())
    data_dir = "data/training"
    num_classes = 25
    batch_size = 40

    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--epochs", required=False, help="the num of epochs file", default=50
    )
    args = parser.parse_args()
    num_epochs = int(args.epochs)
    logger = getLogger(f"logs/classification/{num_epochs}_epochs.txt")
    logger.info(f"Num of epochs given {num_epochs}")

    if not crops_folder_exist(data_dir):
        logger.info(f"No crops found. will create them ...")
        create_data()
        logger.info(f"Crops were created")

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
            os.path.join("crops", "train"), data_transforms["train"]
        ),
        "val": datasets.ImageFolder(
            os.path.join("crops", "val"), data_transforms["val"]
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
    feature_extract = True  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
    model = initialize_model(num_classes, feature_extract)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {str(device)}")

    # Send the model to GPU
    model = model.to(device)

    # Initialize the model to optimize all parameters
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, weight_decay=0.0004, momentum=0.8
    )

    result = update_model_with_saved_checkpoint(check_point_name, model, optimizer)
    start_epoch = -1

    if result:
        logger.info("Found a saved model. will continue from the saved checkpoint")
        model, optimizer, start_epoch = result
        logger.info(f"Start_epoch of saved checkpoint {start_epoch}")
        if num_epochs < start_epoch:
            logger.error(
                f"Saved start_epoch={start_epoch} is bigger than given number of epochs={num_epochs}"
            )
            exit(1)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    # Train and evaluate
    model = train_model(
        logger,
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
    )
