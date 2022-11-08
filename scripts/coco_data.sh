python3 <<HEREDOC
import json
train = open("data/training/coco.json")
test = open("data/testing/coco.json")
trainD = json.load(train)
testD = json.load(test)
print("training images" , len(trainD["images"]))
print("testing images" , len(testD["images"]))
print("total images" , len(testD["images"]) + len(trainD["images"]))
HEREDOC
