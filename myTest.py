from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import json
import cv2
import os
from tqdm import tqdm
import numpy as np


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

# Path
input_path = 'myDataset/test/'

if __name__ == "__main__":
    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # digit 0 ~ 9
    predictor = DefaultPredictor(cfg)
    inferenceResult = []

    count = 0
    for img_name in tqdm(sorted(os.listdir(input_path), key=len)):
        # print(f'Now inference on: {img_name}')
        # if count > 30: break

        img = cv2.imread(input_path + img_name)
        outputs = predictor(img)

        instance = outputs['instances']
        bboxes = instance.get_fields()['pred_boxes'].tensor
        scores = [float(s) for s in instance.get_fields()['scores']]
        pred_classes = [int(s) for s in instance.get_fields()['pred_classes']]
        # print('pred_classes:', pred_classes)
        # print('scores:', scores)
        # print('bboxes:', bboxes)

        # Convert to required COCO format for answer.json
        image_id = int(img_name.replace('.png', ''))
        for score, pred_class, box in zip(scores, pred_classes, bboxes):
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            w = x2 - x1
            h = y2 - y1

            current_detect = {
                "image_id": image_id,
                "score": score,
                "category_id": pred_class,
                "bbox": [x1, y1, w, h]
            }
            inferenceResult.append(current_detect)

        #     caption = '{} {:.3f}'.format(pred_class, score)
        #     draw_caption(img, (x1, y1, x2, y2), caption)
        #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)

        # cv2.imshow('detections', img)
        # cv2.waitKey(0)
        count += 1
    print(f'Finish inferencing on {count} images!')

    # Create answer.json file
    json_object = json.dumps(inferenceResult, indent=4)
    with open("output/answer.json", "w") as outfile:
        outfile.write(json_object)
