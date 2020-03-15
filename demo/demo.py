import sys

sys.path.append("../")

import cv2

from predictor import ClsDemo
from pytorch_classification.config import cfg


config = "../configs/config_cifar10_R50_1gpu.yaml"
img_path = "image/airplane_00003.jpg"
checkpoint_path = "final-model-R50.pth"

cfg.merge_from_file(config)
cfg.merge_from_list(["CHECKPOINT", checkpoint_path])

cls_demo = ClsDemo(cfg)

image = cv2.imread(img_path)
pred = cls_demo.run_on_openv_image(image)
print(pred)