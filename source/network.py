# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import sys
import time

from caffe2.python import workspace

from collections import defaultdict
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import detectron.utils.boxes as box_utils
#import detectron.utils.annotations as ann_utils

class Names:

    classes = [
        'unknown',
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'street',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'hat',
        'backpack',
        'umbrella',
        'shoe',
        'eye accessory',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'plate',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'mirror',
        'dining table',
        'window',
        'desk',
        'toilet',
        'door',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'blender',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush',
        'hair'
    ]

class NeuralNetwork:

    def __init__ (self, cf = '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml', weights = 'https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl'):

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        c2_utils.import_detectron_ops()
        # OpenCL may be enabled by default in OpenCV3; disable it because it's not
        # thread safe and causes unwanted GPU memory allocations.
        cv2.ocl.setUseOpenCL(False)
        # cfg path
        self.cfg = cf
        # weights path
        self.weights = weights
        self.weights = cache_url(self.weights, cfg.DOWNLOAD_CACHE)
        merge_cfg_from_file(self.cfg)
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        self.model = infer_engine.initialize_model_from_cfg(self.weights)
        dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    def detect(self, frame):

        timers = defaultdict(Timer)
        mask_frame = frame
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, mask_frame, None, timers=timers
            )

        # verifica se a imagem deve ser descartada
        boxes, sgms, keypts, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)


        
        bbox_list = []

        if len(boxes) > 0:
            indexes_big = box_utils.filter_big_boxes(boxes, 300)
            indexes_small = box_utils.filter_small_boxes(boxes, 100)
            for i in range(len(boxes)):
                if (i in indexes_big and i in indexes_small):
                    if classes[i] in [1,2,3] and boxes[i, 4] > 0.7:
                        box = boxes[i]
                        bbox_list.append([int(box[0]),int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])], classes[i])  


#        mask_frame = vis_utils.vis_one_image_opencv(mask_frame, cls_boxes, cls_segms, cls_keyps, thresh=0.8, kp_thresh=2,
#        show_box=True, dataset=CocoNames, show_class=True) #, hiden_indexes=True, indexes_shown=[1])

        return bbox_list