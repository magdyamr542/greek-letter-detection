"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""
import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import json
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import frcnn.transforms as T
from frcnn.engine import train_one_epoch, evaluate
import frcnn.utils as utils
from constants import category_to_label


model_name = "model_detection.pt"


def update_model_with_saved_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


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
            (1000, round(img.size[1] * 1000.0 / float(img.size[0]))), Image.BILINEAR
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


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = len(category_to_label)
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

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8, weight_decay=0.0004)
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85)

    model, optimizer, start_epoch = update_model_with_saved_checkpoint(
        model_name, model, optimizer
    )

    num_epochs = 50
    for epoch in range(start_epoch + 1, num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        if epoch < 4:
            lr_scheduler1.step()
        elif epoch > 40 and epoch < 48:
            lr_scheduler2.step()
        evaluate(model, data_loader_test, device=device)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_name,
        )

    print("That's it!")


main()
