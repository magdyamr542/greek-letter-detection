import os
from random import choices
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from constants import model_input_size

def main(data_dir: str):

    allClasses = os.listdir(os.path.join(data_dir, "images", "training"))
    currentClasses = os.listdir(os.path.join(data_dir, "crops", "train"))
    restClasses = list(set(allClasses) - set(currentClasses))

    numRestClassesToCreate = 1000 - len(currentClasses)
    print(f"remaing crops {numRestClassesToCreate}")

    # pick num_classes labels (dirs)
    usedClasses = choices(restClasses, k=numRestClassesToCreate)

    done = 0
    for usedClass in usedClasses:
        print(f"creating crops for {usedClass}")

        allTrainImages = glob.glob(
            os.path.join(data_dir, "images", "training", f"{usedClass}/*.png")
        )

        print(f"all images for {usedClass}", len(allTrainImages))
        train, val = train_test_split(allTrainImages, random_state=4)
        print(f"training images for {usedClass}", len(train))
        print(f"validation images for {usedClass}", len(val))

        for valImg in val:
            imgDir = valImg.split("/")[-2]
            imgName = valImg.split("/")[-1]
            if not os.path.exists(os.path.join(data_dir, "crops", "val", imgDir)):
                os.makedirs(os.path.join(data_dir, "crops", "val", imgDir))

            try:
                im = Image.open(valImg).convert("RGB")
                im = im.resize((model_input_size, model_input_size))
                im.save(os.path.join(data_dir, "crops", "val", imgDir, imgName))
            except:
                print("error while saving image", valImg)

        for trainImg in train:
            imgDir = trainImg.split("/")[-2]
            imgName = trainImg.split("/")[-1]
            if not os.path.exists(os.path.join(data_dir, "crops", "train", imgDir)):
                os.makedirs(os.path.join(data_dir, "crops", "train", imgDir))

            try:
                im = Image.open(trainImg).convert("RGB")
                im = im.resize((model_input_size, model_input_size))
                im.save(os.path.join(data_dir, "crops", "train", imgDir, imgName))
            except:
                print("error while saving image", trainImg)
        done += 1
        print("left" , len(usedClasses) - done)

if __name__ == "__main__":
    main("chinese-data")
