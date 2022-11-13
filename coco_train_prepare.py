from argparse import ArgumentParser
import glob
from pylabel import importer
import json
import os
from pathlib import Path
import shutil

# maps a category number to an index char
coco_map = {
    7: 0,
    8: 1,
    9: 2,
    14: 3,
    17: 4,
    23: 5,
    33: 6,
    45: 7,
    59: 8,
    77: 9,
    100: 10,
    107: 11,
    111: 12,
    119: 13,
    120: 14,
    144: 15,
    150: 16,
    161: 17,
    169: 18,
    177: 19,
    186: 20,
    201: 21,
    212: 22,
    225: 23,
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--yolo-path",
        required=True,
        help="path of the directory that contains the yolo directory",
    )
    args = parser.parse_args()
    yolo_path = args.yolo_path

    print(f"Will create the yolo data in {yolo_path} ...")
    Path(yolo_path).mkdir(parents=True, exist_ok=True)

    # create the dirs
    training_dir = os.path.join(yolo_path, "training")
    images_dir = os.path.join(training_dir, "images")
    labels_dir = os.path.join(training_dir, "labels")

    print(f"Creating images directory {images_dir} ...")
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    print(f"Creating labels directory {labels_dir} ...")
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    # fill images dir
    cocoJson = get_coco()
    print(f"num of training images {len(cocoJson['images'])}")

    # cp images
    print(f"copying the images from to {images_dir} ...")
    create_images(images_dir, cocoJson)

    # create coco file
    coco_file_path = os.path.join(training_dir, "coco.json")
    print(f"creating coco.json file in {coco_file_path}")
    create_coco_file(cocoJson, coco_file_path)

    # fill labels dir
    create_labels(coco_file_path, labels_dir)

    # cp the dataset.yml file
    # dataset_yaml_file = os.path.join("yolo", "dataset.yaml")
    # dataset_yaml_new_file = os.path.join(training_dir, "dataset.yaml")
    # print(f"Copying {dataset_yaml_file} to {training_dir}")
    # shutil.copyfile(dataset_yaml_file, dataset_yaml_new_file)


def create_coco_file(cocoJson, filePath):
    with open(filePath, "w") as f:
        json.dump(cocoJson, f)


def create_images(images_dir_path, cocoJson):
    imagesjpg = glob.glob(os.path.join("data", "training", "images", "**/**/*.jpg"))
    imagesJPG = glob.glob(os.path.join("data", "training", "images", "**/**/*.JPG"))
    images = imagesjpg + imagesJPG
    coco_images = cocoJson["images"]
    coco_image_names = set(list(map(lambda object: object["file_name"], coco_images)))
    for imagePath in images:
        imageName = imagePath.split("/")[-1]
        if imageName in coco_image_names:
            newImagePath = os.path.join(images_dir_path, imageName)
            shutil.copyfile(imagePath, newImagePath)


def create_labels(coco_file_path, label_dir_path, images_dir_path=""):
    dataset = importer.ImportCoco(
        coco_file_path, path_to_images=images_dir_path, name="BCCD_coco"
    )
    print(f"Creating the labels in {label_dir_path}")
    dataset.export.ExportToYoloV5(label_dir_path)[0]

    # post process the dataset
    print(f"Updating the created labels in {label_dir_path}")
    labelsFiles = glob.glob(os.path.join(label_dir_path, "*.txt"))
    for labelFile in labelsFiles:
        with open(labelFile, "r+") as f:
            new_file_lines = []
            for line in f:
                nums = line.split(" ")
                if coco_map.get(int(nums[0]), None) is not None:
                    nums[0] = str(coco_map[int(nums[0])])
                    new_file_lines.append(" ".join(nums))

        with open(labelFile, "w+") as f:
            for line in new_file_lines:
                f.write(line)


def get_coco():
    path = "./data/training/coco.json"
    f = open(path)
    cocoJson = json.load(f)
    images = []
    for img in cocoJson["images"]:
        img["file_name"] = img["file_name"].split("/")[-1]
        img["img_url"] = img["img_url"].split("/")[-1]
        images.append(img)
    cocoJson["images"] = images
    f.close()
    return cocoJson


if __name__ == "__main__":
    main()
