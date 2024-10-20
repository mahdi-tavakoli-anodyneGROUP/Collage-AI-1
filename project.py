import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.2.103 -q

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

!rm -rf /content/datasets

!mkdir -p {HOME}/datasets
%cd {HOME}/datasets


!pip install roboflow --quiet

import roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="BhJ63IemG3ACRXXxUyzC")
project = rf.workspace("dental-decay").project("check-hfbzl-zemwx")
version = project.version(21)
dataset = version.download("yolov11")

import yaml

with open(f"{dataset.location}/data.yaml", 'r') as f:
    dataset_yaml = yaml.safe_load(f)
dataset_yaml["train"] = "../train/images"
dataset_yaml["val"] = "../valid/images"
dataset_yaml["test"] = "../test/images"
with open(f"{dataset.location}/data.yaml", 'w') as f:
    yaml.dump(dataset_yaml, f)

%cd {HOME}

!yolo task=segment mode=train model=yolov8s-seg.pt data={dataset.location}/data.yaml epochs=75 imgsz=640 optimizer=SGD

!ls {HOME}/runs/segment/train/

%cd {HOME}
Image(filename=f'{HOME}/runs/segment/train/val_batch0_pred.jpg', width=600)

%cd {HOME}

!yolo task=segment mode=val model={HOME}/runs/segment/train/weights/best.pt data={dataset.location}/data.yaml

%cd {HOME}
!yolo task=segment mode=predict model={HOME}/runs/segment/train/weights/best.pt conf=0.5 source={dataset.location}/test/images save=true