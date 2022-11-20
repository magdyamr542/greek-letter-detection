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

1. the metric and logs are saved under `./sacred_test_classification` and `./sacred_train_classification` directories
1. The experiments are tracked
1. Commands
	1. Testing `python3 test_classification.py with checkpoint=<chkpt path>`
	1. Training `python3 classification.py with checkpoint=<chkpt path> epochs=<num epochs>`

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

### Training with chinese dataset
- To apply transfer learning a chinese dataset was used from [here](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets)
- The dataset was huge so a fraction of it was used
	- Number categories `800`
	- Number of training images `348176`
	- Number of testing images `83972`
	- categories are:
		```
		'#'             ф╣▒   х░┤   хдЗ   хи▒   х▓м   хР▒   х░Х   хЭО   цбз   цЕО   цКФ   ц╕п   цЦЧ   цЮБ   ч▓в   ч╝з   члп   чПЩ   чХФ   чЫЧ   шБЛ   шДШ   шИЫ   шМЬ   шРО   шЫЖ   щ╕╣   ще╢   щлж   щУТ   щЧ╛
		-              ф╗б   х▒║   хе│   х╕И   хМ┐   хРн   хХЦ   хЭЬ   цбО   ц╜Ж   ц┤л   ц╝п   цЧГ   цЮД   ч╝В   чЗГ   чЛЧ   чПЮ   чХЪ   чЬ╡   шбо   шДЮ   шИЯ   ш░н   шСЧ   шЫЗ   щ╗╗   щЕ╢   щлМ   щУЦ   щЧ╝
		'['             ф╗д   х▓╡   хЕ╝   х▒И   х║н   х┐С   х║Ц   хЭЭ   цвЕ   цжЙ   цЛ╜   цп│   цЧж   цЮЬ   чвб   чзн   чЛЮ   чР│   чЦГ   чЬН   ш╕в   ш░е   ш░й   шН╖   шТО   шЫй   щ╝╜   щЕб   щлШ   щУШ   щШ╝
		'`'             ф╝д   х▓▒   хЕл   хИ┐   х╝н   х▓С   хЦБ   х╣Ю   цвж   цЖЦ   цЛ╝   цп╜   цЧЖ   ц▓Я   чвМ   чзС   члЯ   чР░   чЦд   ч╗Э   ш╛В   ш╕Е   шЙ│   шНГ   шТп   шЫК   щ▒┐   щЕН   щ╝м   щФ╖   щШЛ
		'}'             ф║ж   хАб   хеШ   хиЗ   хНГ   хС╡   хЦК   хЮа   ц│г   цжШ   цЛВ   цп╡   цЧн   цЯГ   чвЫ   ч╜и   чм║   чРв   чЦП   чЭ╜   ш╣В   ш╢Е   шЙ┐   шНн   шТЧ   ш┐Ь   щ▓╝   щ╕ж   щмп   щФ╝   щШМ
		0              ф╗З   хАй   хЕЪ   хий   хнд   хСФ   хЦП   хЮЕ   цг╡   цжЬ   цЛг   ц╡П   цЧо   цЯе   чВЬ   ч╝И   чМ╕   чРГ   ч▓Ч   чЭб   шВ┐   шЕФ   ш┤к   шНТ   шТЯ   шЬ╖   щ▓▓   щ╣ж   щ▓н   щФв   щШФ
		colon          ф╕И   хак   хж╗   хиЙ   хнж   х┐Т   хЦЩ   хЮж   цГ┤   ц▓з   цЛЖ   ц▓П   цЧТ   ч║┤   чГИ   чиЖ   чмд   чРЗ   чЧ▒   чЭг   швл   ш╛Ж   ш╛К   шНЫ   ш░У   шЬг   щ╝а   щЖ╡   щ╝Н   щФз   щ╛Щ
		H              ф╝л   хАУ   хж▓   хИО   хНО   хТ┐   хЦЭ   хЮЖ   цГ╢   цЗ╡   цЛИ   цП╢   ц░Ш   ч║╖   чГл   чИЖ   чмк   чРп   чЧк   чЭж   ш╖г   ш╡Ж   шК╕   шНЮ   шУ┐   шЬГ   щА╡   щЖг   щнБ   щФи   щЩД
		J              ф║н   хАЩ   хЖ╢   хиШ   хНЮ   хТА   хЧД   хЮн   цГз   цзБ   цЛК   цПй   цШЭ   ч╕╗   чГн   чиО   чмП   чРР   чЧХ   чЭР   ш╡г   шЖА   шКе   ш░о   шУж   шЬе   щАЖ   щЖЙ   щнЕ   щФй   щЩз
		V              ф╗О   хБе   хЖ░   х╖й   х┤о   хТВ   хЧЦ   ц┤▒   цги   цЗИ   ц│М   цпп   ц░Щ   ч╗║   чГШ   чЙ┐   чМУ   чРЫ   чШ┤   чЭЫ   ш░г   шЖг   шКи   шо╛   шУЙ   шЬЬ   щАЛ   щЖн   щнЙ   щФЛ   щЩР
		vertical_bar   ф┐П   хбН   хЖ▒   х╜й   хо┤   хТй   х╜Ш   ц║╖   цгЛ   цзЪ   ц╖М   цпФ   цЩ║   ч╗╝   ч╗д   чЙ║   чмЫ   чСХ   чШБ   ч╛Ю   шг╕   шЖИ   шкЙ   шо╜   шУУ   шЬЮ   щАЫ   щжР   щНк   щФО   щЪ│
		тА░            ф╣Р   х╣в   хжВ   х╖Й   хоГ   хТм   хШ▒   ц╖▒   цГм   цИ╛   цМб   ц▓Р   цЩж   ч╗╣   чД▒   чЙ▓   чн╖   чСЫ   чШЧ   чЮЕ   шГА   шЖК   шкК   шОЖ   шФ╕   ш░Э   щ│б   щ╣и   щ╕п   щФЪ   щЪ╝
		тВз            ф║Т   х░в   хжд   хЙб   ход   хТМ   хШО   ц╝╢   цГп   циК   цМЫ   цРа   ц╕Ъ   ч╗░   ч▓е   чЙВ   чн╡   чТЬ   чШЩ   чЮН   шГБ   ш╜з   шКП   шоз   шФ╣   шЭа   щБВ   щ╣И   щ│Р   щ╣Х   щЪ╢
		тИ┤            ф╕У   хвЕ   хжЖ   хЙЕ   хоЕ   х╜У   хШЮ   ц╝╣   цгЪ   цИР   ц░Н   цРн   цЪЕ   ч╛╝   че║   чйД   ч▓о   чФ▒   ч│Щ   чЯз   шГД   ш┐З   ш╜л   ш┤п   шФУ   шЭд   щвА   щй║   щ╝Т   щ▓Х   щЪ░
		тИ╡            ф╛Ф   хВЕ   хжК   хйз   хОН   х╝У   хШЯ   ц░┤   ц▓д   цИЧ   цН╗   цРР   ц╣Ы   ч╛▓   че╕   чЙп   ч╝О   чФ▓   чЩ╗   ш┤┤   шГЖ   ш╡З   ш╡л   ш░п   ш┐Ш   шЭИ   щвЕ   щй╖   щТ╛   щХ░   щЫ│
		тИе            ф╝Ъ   хВй   хжп   хЙй   хп║   хУА   хЩА   ц░▓   цДж   циЯ   цНВ   цРЬ   цЫЫ   ч╝┤   че╝   чЙТ   чОЗ   чФн   чЩ╛   ш╛╢   шги   ш░З   ш░Л   шп╖   шЩ║   шЭО   щвЗ   щй╗   щТ▒   щХа   щЫБ
		тЦ│            ф╝Ю   хвЯ   х│з   хйк   х╝П   хУБ   хЩм   ц│а   цДЙ   ц╕й   цНЕ   ц┐С   ц▒Ь   ч╝║   ч│Е   ч│К   чОЙ   чФН   чЩЮ   ш╜┤   шГи   шЗ╗   шЛа   шп╛   шЩ╝   шЮ╡   щВм   щй╣   щТг   щХА   щЫк
		тЧО            ф┐Я   х▒г   х┤з   хЙН   х▒П   хУк   хЩН   ц┐а   цДЯ   цй╣   цнЖ   цСз   цЬн   ч╝▒   чеБ   чК╣   чоН   чФЬ   чЪЗ   ш╜▒   шГЩ   шзД   шЛд   шП╝   ш▒Ъ   шЮа   щВп   щй░   щТе   щХб   щЫп
		уАБ            х┐╡   хг░   х╝з   хЙХ   хП╢   х║Ф   х╕Ъ   ц╕а   ц╗е   цЙ╝   цНМ   цСФ   ц╕Э   ч░┐   чЕд   чкЕ   чОЯ   ч│Х   чЪЛ   ш╢╛   шГЮ   шзе   шЛЬ   шП▒   шЪ║   шЮИ   щвЧ   щйм   щТЕ   щХд   щ│Ь
		уОЬ            х║╕   хГ│   хЗ╡   хЙЬ   хпЗ   х╣Ф   хЪ╖   цА╗   це╣   цйж   цНЯ   цТД   цЭб   ч▒╗   чеУ   чкм   чп╝   ч╗Х   чЫ╕   ш░▓   ш╢Д   шзИ   ш╕м   шПб   шЪг   шЮп   щвЭ   щк╝   щТи   щХе   щЬ╛
		ф║╗            х║╢   хГЗ   хЗ░   хк╕   хпМ   хФ│   хЪЖ   цаЗ   ц╡Е   цйЗ   цо│   ц▒Ф   цЭз   чА╡   чеЧ   чкТ   ч╗П   чХ┐   чЫ╝   ш╡а   шД╛   ш╣И   шМ╕   шПЗ   шЪм   щ╕╖   щ╣г   щкД   щ│У   щХи   щЬ╣
		ф╕║            х╖╖   хгм   хЗА   х░К   хПН   хФ╛   х┤Ы   цАл   ц░Е   цЙй   цОИ   цФо   цЭМ   чАг   чжА   ч║л   чПа   чХ╕   чЫ▒   шАЖ   шдб   шИА   шМ╝   шпй   шЫ╛   щ╕╜   щГИ   щкС   щУ░   щХЙ   щЭ╝
		ф╕╢            х╖╛   хд╝   хзЮ   хК▓   хпУ   хФ╝   хЫд   ц│б   цеа   ц╕К   цОТ   цФТ   цЭО   чаЦ   чЖК   ч╝л   чпЖ   чХж   чЫ▓   шаХ   шДм   шИВ   шМА   шпУ   шЫ▓   щ╕╝   щгФ   щ║Л   щУА   щ╣Ч   щЭв
		ф╝╛            х╛╡   х╝Д   х╖и   хКЭ   хПЧ   хФа   хЬй   ц╛б   цеж   цКа   цоЦ   цХХ   ц┤Ю   чаЭ   чЖЯ   чЛД   чпК   чХИ   чЫЖ   ш╛Б   шДП   шИД   шМБ   шРГ   шЫА   щ╕╡   щ╕д   щ▓Л   щУи   щЧ┐   щЮН
		ф╝▓            х╣┐   хДЖ   хи╝   х┐л   хР╡   хФг   хЬп   ц▓Б   цЕи   цкР   цоЪ   цЦУ   ц╗Ю   ч╣Б   ч╛з   члЛ   чПР   чХП   чЫР   шБ┐   шдУ   шИЕ   шМС   шРж   шЫД   щ╕╢   ще╜   щлА   щУй   щЧ╗   щ║Я
		```
	- The classification model was trained in 50 epochs and the statistics were
		```
		train Loss: 0.0789 Acc: 0.9882
		val Loss: 0.0950 Acc: 0.9799
		Training complete in 670m 2s
		```
	- The classification model was tested and the results were perfect with accuracy almost 1.0 (probably overfitting but anyway)
	#### Transfer learning with chinese dataset
	- the weights of the chinese trained model were saved and used as initialization weights to train the actual model
	- the results didn't differ that much from the results obtained while using the `ImageNet` weights
		```
		train Loss: 0.3699 Acc: 0.8907
		val Loss: 0.7304 Acc: 0.7954
		```
	- the accuracy was `0.555084`
