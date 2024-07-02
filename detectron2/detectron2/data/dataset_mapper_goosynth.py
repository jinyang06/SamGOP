# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from . import detection_utils as utils
from . import transforms as T
import cv2
from PIL import Image, ImageDraw
from detectron2.structures import BoxMode
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            use_keypoint: bool = False,
            instance_mask_format: str = "polygon",
            keypoint_hflip_indices: Optional[np.ndarray] = None,
            precomputed_proposal_topk: Optional[int] = None,
            recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk = precomputed_proposal_topk
        self.recompute_boxes = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # ims = Image.fromarray(image)
        # ims.show()
        utils.check_image_size(dataset_dict, image)
        width, height = dataset_dict['width'], dataset_dict['height']
        gaze_related_ann = dataset_dict['gaze_related_ann']
        gaze_related_ann['gaze_cx'], gaze_related_ann['gaze_cy'] = gaze_related_ann['gaze_cx'], \
                                                                   gaze_related_ann['gaze_cy']
        gaze_related_ann['hx'], gaze_related_ann['hy'] = gaze_related_ann['hx'], gaze_related_ann['hy']

        ##########################################################
        segmentation_annotations_list = []
        segmentation_annotations = dataset_dict['annotations']
        for index, item in enumerate(segmentation_annotations):
            item['bbox'] = np.array([item['bbox']])
            item['bbox'][:, 2] = item['bbox'][:, 0] + item['bbox'][:, 2]
            item['bbox'][:, 3] = item['bbox'][:, 1] + item['bbox'][:, 3]
            item['bbox_mode'] = BoxMode.XYXY_ABS
            item['segmentation'] = np.array(item['segmentation']).reshape(int(len(item['segmentation'][0]) / 2), 2)
            item['segmentation'] = item['segmentation'] / [640, 480]
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list

        segmentation_annotations_list = []
        for index, item in enumerate(segmentation_annotations):
            box = item['bbox']
            box[:, [0, 2]] = box[:, [0, 2]] * 224 / width
            box[:, [1, 3]] = box[:, [1, 3]] * 224 / height
            item['bbox'] = box.astype(np.float64)

            seg = item['segmentation']
            seg[:, 0] = seg[:, 0] * 224
            seg[:, 1] = seg[:, 1] * 224
            seg = seg.astype(np.float64)
            seg = seg.flatten()
            item['segmentation'] = [seg]
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list

        new_segmentation_annotations = []
        for index, item in enumerate(segmentation_annotations):
            box = item['bbox']
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > 224] = 224
            box[:, 3][box[:, 3] > 224] = 224
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box = np.array(box, dtype=np.float64)

            item['segmentation'][0][item['segmentation'][0] > 224] = 224
            item['segmentation'][0][item['segmentation'][0] < 0] = 0

            if box.shape[0] != 0:
                box = box[0]
                item['bbox'] = box
                new_segmentation_annotations.append(item)
        dataset_dict['annotations'] = new_segmentation_annotations

        annos = new_segmentation_annotations

        ##########################################################

        gaze_box = gaze_related_ann['gaze_box']
        gaze_box = np.array([gaze_box])
        gaze_box = gaze_box.astype(np.int32)
        # head_box = gaze_related_ann['head_box']
        # head_box = np.array([head_box])
        # head_box = head_box.astype(np.int32)

        eye = [float(gaze_related_ann['hx']) / 640, float(gaze_related_ann['hy']) / 480]
        gaze = [float(gaze_related_ann['gaze_cx']) / 640, float(gaze_related_ann['gaze_cy']) / 480]
        img = Image.open(dataset_dict['file_name'])
        img = img.convert('RGB')
        gaze_x, gaze_y = gaze
        eye_x, eye_y = eye

        # crop head
        k = 0.1
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        gaze_p = np.array([gaze_x, gaze_y])
        eye_p = np.array([eye_x, eye_y])
        gaze_direction = gaze_p - eye_p

        if gaze_direction.mean() != 0:
            gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)

        dataset_dict['gaze_direction'] = gaze_direction
        dataset_dict['gaze'] = [gaze_x, gaze_y]
        dataset_dict['eye'] = [eye_x, eye_y]
        dataset_dict['gaze_related_ann']['gaze_cx'] = gaze_x
        dataset_dict['gaze_related_ann']['gaze_cy'] = gaze_y
        dataset_dict['gaze_related_ann']['hx'] = eye_x
        dataset_dict['gaze_related_ann']['hy'] = eye_y
        dataset_dict['gaze_related_ann']['gaze_box'] = gaze_box
        # dataset_dict['gaze_related_ann']['head_box'] = head_box
        # head_channel = utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
        #                                           resolution=224, coordconv=False).unsqueeze(0)
        # dataset_dict['head_channel'] = head_channel
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face = face.resize((224, 224), Image.BICUBIC)
        face = np.array(face)
        face = torch.as_tensor(np.ascontiguousarray(face.transpose(2, 0, 1)))
        dataset_dict['face'] = face
        # face = self.transform(face)
        img = img.resize((224, 224), Image.BICUBIC)
        im = img
        img = np.array(img)
        ims = img
        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        dataset_dict['image'] = img

        gaze_heatmap = torch.zeros(64, 64)  # set the size of the output
        gaze_heatmap = utils.draw_labelmap(gaze_heatmap, [gaze_x * 64, gaze_y * 64],
                                           3,
                                           type='Gaussian')
        dataset_dict['gaze_heatmap'] = gaze_heatmap

        #########################
        # visualization
        # im = Image.fromarray(im)
        # draw = ImageDraw.Draw(im)
        # draw.ellipse([gaze_x*224 - 2, gaze_y*224 - 2, gaze_x*224 + 2, gaze_y*224 + 2],fill='red', outline=None)
        # draw.ellipse([eye_x*224 - 2, eye_y*224 - 2, eye_x*224 + 2, eye_y*224 + 2], fill='red',outline=None)
        # draw.line([eye_x*224, eye_y*224, gaze_x*224, gaze_y*224],fill='yellow',width=2)
        # im.show()
        # x=0

        # draw = ImageDraw.Draw(im)
        # for item in annos:
        #     bbox = item['bbox'].astype(int)
        #     bbox = bbox.tolist()
        #     draw.rectangle(bbox, fill=None, outline='green', width=3)
        # # draw.rectangle(hbox.tolist(), fill=None, outline='red', width=3)
        # # draw.rectangle([eye_x * 224 - 40, eye_y * 224 - 40, eye_x * 224 + 40, eye_y * 224 + 40],fill=None, outline='red', width=3)
        # draw.ellipse([gaze_x*224 - 2, gaze_y*224 - 2, gaze_x*224 + 2, gaze_y*224 + 2],fill='red', outline=None)
        # draw.ellipse([eye_x*224- 2, eye_y*224 - 2, eye_x*224 + 2, eye_y*224 + 2], fill='red',outline=None)
        # draw.line([eye_x*224, eye_y*224, gaze_x*224, gaze_y*224],fill='yellow',width=2)
        # # im_with_hm = utils.generate_attention_map(im, gaze_heatmap)
        # # im_with_hm.show()
        # im.show()
        # x=0

        #########################
        # image_shape = (224, 224)

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        # if self.proposal_topk is not None:
        #     utils.transform_proposals(
        #         dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
        #     )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        # if "annotations" in dataset_dict:
        #     self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
