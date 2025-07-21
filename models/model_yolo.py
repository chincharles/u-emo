from torch import nn
from models.yolo_utils import prepare_yolo, rescale_boxes, non_max_suppression
from torchvision import transforms
import numpy as np
import torch


def load_YOLO(args):
    print(args.model_path)
    yolo = prepare_yolo(args.model_path)
    return yolo


class YoloBaseModel(nn.Module):
    def __init__(self, args):
        super(YoloBaseModel, self).__init__()
        self.yolo = load_YOLO(args)

    def prepare_data_by_yolo(self, data):

        bbox_yolo = self.get_bbox(data)

        if bbox_yolo is None:
            pass
        elif len(bbox_yolo) == 0:
            pass
        else:
            bbox_yolo = bbox_yolo[0]
            test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            image_body = data[bbox_yolo[1]:bbox_yolo[3], bbox_yolo[0]:bbox_yolo[2]]
            if 0 not in image_body.shape:
                image_body = test_transform(data)
                return image_body

        test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_body = test_transform(data)
        return image_body


    def get_bbox(self, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):

        image_yolo = image_context
        with torch.no_grad():
            detections = self.yolo(image_yolo)
            nms_det = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
            if nms_det is None:
                return None
            det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))

        bboxes = []
        for x1, y1, x2, y2, _, _, cls_pred in det:
            if cls_pred == 0:  # checking if predicted_class = persons.
                x1 = int(min(image_context.shape[1], max(0, x1)))
                x2 = int(min(image_context.shape[1], max(x1, x2)))
                y1 = int(min(image_context.shape[0], max(15, y1)))
                y2 = int(min(image_context.shape[0], max(y1, y2)))
                bboxes.append([x1, y1, x2, y2])
        return np.array(bboxes)

    def forward(self, data):
        self.yolo.eval()
        return self.prepare_data_by_yolo(data)


