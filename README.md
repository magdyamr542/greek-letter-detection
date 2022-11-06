## OCR project for the detection of greek letters in historical documents

[Task](https://lme.tf.fau.de/competitions/icfhr2022-competition-on-detection-and-recognition-of-greek-letters-on-papyri/)

### Getting started

1. Create `./assets` directory ,download the [dejavu fonts](https://sourceforge.net/projects/dejavu/) and put the fonts in the `./assets` directory
1. Download the data from [here](https://lme.tf.fau.de/competitions/icfhr2022-competition-on-detection-and-recognition-of-greek-letters-on-papyri/)
   - create a `data` directory with the following structure
   ```
   	data
   	├── cropped # is auto created while detection
   	├── testing # the testing data
   	│   └── images
   	│       └── homer2
   	└── training # the training data
   		├── classification_model.pt # can be copied externally  from where it was trained
   		├── coco.json
   		├── crops # is auto generated while classification training
   		│   ├── train
   		│   └── val
   		├── image_list.bin # is auto generated while classification training
   		└── images
   			└── homer2
   ```
1. Create a `venv` with `virtualenv` and install the dependencies:
   - `source venv/bin/activate`
   - `pip install -r requirements.txt`


### Classification

1. the metric and logs are saved under `./logs/classification` directory
1. Every directory there represents the log files for the number of test images used while training
1. Under `testing` directory we save the accuracy of the model
1. While training, the log files are saved with this format. `[num epochs]_epochs.txt` which means that these logs are the result when we trained the model with `[num epochs]` epochs

### Sample commands

1. Testing classification model
	1. `python3 test_classification -c data/training/classification_check_points/classification_model_check_point_not_pretrained_epoch_100.pt`
	1. It's important for the checkpoint to contain the number of epochs
1. Testing detection model
   1. `python3 test_detection.py -f data/testing/images/homer2/txt8/P_Laur_IV_128r.jpg`

### Train YOLO
1. Clone the yolo project and install the requirements
   1. `git clone https://github.com/ultralytics/yolov5  # clone`
   2. `cd yolov5`
   3. create a **virtualenv** to install the dependencies
   4. `pip install -r requirements.txt`
1. Create a directory next to the `yolov5` directory and call it `training`
   1. cd `..`
   1. `mkdir training`
   1. `cd training`
   1. the directory should contain `images` directory ,`labels` directory and `dataset.yaml` file
	1. The `images` directory contains the training images 
	1. The `labels` directory contains a `[image name].txt` file for each image describing the bounding boxes for the image
	1. The `labels` directory can be created using the `./coco2yolov5.ipynb` file
	1. The `dataset.yaml` file can be found in `./yolo/dataset.yaml`
1. Train the yolo model
	1. `cd yolov5`
	1. copy the `./yolo/dataset.yaml` to the `data` directory inside of yolo
	1. `python3 train.py --img 640 --batch 16 --epochs 3 --data dataset.yaml --weights yolov5s.pt`