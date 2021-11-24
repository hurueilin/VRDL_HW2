# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import json
import cv2


def get_SVHN_dict(train_labels):
    dataset_dicts = []

    for label in train_labels:
        filename = './myDataset/train/' + label['filename']
        height, width = cv2.imread(filename).shape[:2]
        record = {}
        record['file_name'] = filename
        record['image_id'] = label['filename']
        record['height'] = height
        record['width'] = width

        objs = []
        for obj in label['boxes']:
            bbox = list(map(int, [obj['left'], obj['top'], obj['width'], obj['height']]))

            if int(obj['label']) == 10:  # convert the label of digit '0'
                class_name = 0
            else:
                class_name = int(obj['label'])

            obj_dict = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": class_name,
            }
            objs.append(obj_dict)
            record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts  # list of dicts

# Load raw training labels
with open('myDataset/digitStruct.json') as f:
    train_labels = json.load(f)

# Registering the Dataset
print('Registering dataset...')
for d in ['train']:
    DatasetCatalog.register("SVHN_"+d, lambda d=d: get_SVHN_dict(train_labels))
    MetadataCatalog.get("SVHN"+d).thing_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# train_dataset = DatasetCatalog.get("SVHN_train")
train_dataset_metadata = MetadataCatalog.get("SVHN_train")
print('Finish registering dataset.')


# Visualizing the Train Dataset
# from detectron2.utils.visualizer import Visualizer
# import random
# dataset_dicts = get_SVHN_dict(train_labels)
# for d in random.sample(dataset_dicts, 30):  # Randomly choose images from the Set
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=train_dataset_metadata)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("caption", vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)


# Train
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("SVHN_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = 'model_final_280758(R50-FPN).pkl'
cfg.MODEL.WEIGHTS = 'model_final_f6e8b1(R101-FPN).pkl'
cfg.SOLVER.IMS_PER_BATCH = 2  # batch size
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 50000  # (default: 40000)  one epoch = 16701 iters (33402[# of training images] / 2[batch size])
cfg.SOLVER.STEPS = (5000,)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # digit 0 ~ 9

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
