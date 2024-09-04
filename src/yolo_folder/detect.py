# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import functools
from pathlib import Path

import numpy as np
import torch
import yaml
import pickle

from src.yolo_folder.models.common import DetectMultiBackend
from src.yolo_folder.utils.dataloaders import LoadImages
from src.yolo_folder.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from src.yolo_folder.utils.torch_utils import select_device, smart_inference_mode


@functools.lru_cache(1)
def _init_detector(
        weights,
        imgsz,
        bs,
        device,
        half,
) -> torch.nn.Module:
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    model.warmup(imgsz=(1 if model.pt or model.triton else bs, 3, *imgsz))  # warmup

    return model


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


@smart_inference_mode()
def run(
        weights,
        source,
        classes='C:/test/recognition_core/models/detection/yolov5l_12804/classes.yml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.3,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
):
    # Load model
    bs = 1
    model = _init_detector(weights, imgsz, bs, device, half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    annotations = {
        'annotations': [],
        'categories': {},
        'images': [],
    }
    for image_id, (path, im, im0s, vid_cap, s) in enumerate(dataset, 1):
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        annotations['images'].append(
            {
                'id': image_id,
                'file_name': Path(path).name,
                'width': im0s.shape[1],
                'height': im0s.shape[0],
            }
        )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    cls += 1
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    xywh = [int(x) for x in xywh]
                    annotations['annotations'].append(
                        {
                            'id': len(annotations['annotations']) + 1,
                            'bbox': xywh,
                            'area': xywh[2] * xywh[3],
                            'category_id': cls,
                            'image_id': image_id,
                        }
                    )
                    annotations['categories'][cls] = {
                        'id': cls,
                        'name': cls - 1,
                        'supercategory':'root',
                    }

    annotations['categories'] = list(annotations['categories'].values())

    if classes is not None:
        if isinstance(classes, (str, Path)):
            with open(classes, "rb") as f:
                classes = yaml.safe_load(f)['names']
                # classes = pickle.load(f)

        for c in annotations['categories']:
            cls = c['name']
            c['name'] = classes[cls]

    return annotations

