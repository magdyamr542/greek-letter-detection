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

### Sample commands

1. `python3 test_detection.py -f data/testing/images/homer2/txt8/P_Laur_IV_128r.jpg`
1. `python3 test_classification.py -f data/cropped/30.png`
