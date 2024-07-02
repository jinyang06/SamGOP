# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging

import numpy as np
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, PolygonMasks

from pycocotools import mask as coco_mask
import cv2
from PIL import Image,ImageDraw
import torchvision.transforms.functional as TF
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]

def compute_iou(box1, box2):
    intersection_left = np.maximum(box1[0], box2[0])
    intersection_right = np.minimum(box1[2], box2[2])
    intersection_width = np.maximum(0, intersection_right - intersection_left)

    intersection_top = np.maximum(box1[1], box2[1])
    intersection_bottom = np.minimum(box1[3], box2[3])
    intersection_height = np.maximum(0, intersection_bottom - intersection_top)

    intersection_area = intersection_width * intersection_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        iou = 0.0
    else:
        iou = intersection_area / union_area

    return iou


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                # vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),

        T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
        T.RandomContrast(intensity_min=0.5, intensity_max=1.5),
        T.RandomSaturation(intensity_min=0, intensity_max=1.5)
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            tfm_gens,
            image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        width, height = dataset_dict['width'], dataset_dict['height']
        gaze_related_ann = dataset_dict['gaze_related_ann']
        gaze_related_ann['gaze_cx'], gaze_related_ann['gaze_cy'] = gaze_related_ann['gaze_cx'] / 640 * 1920, \
                                                                   gaze_related_ann['gaze_cy'] / 480 * 1080
        gaze_related_ann['hx'], gaze_related_ann['hy'] = gaze_related_ann['hx'] / 640 * 1920, gaze_related_ann[
            'hy'] / 480 * 1080
        # gaze_id = gaze_related_ann['gazeIdx']
        ##########################################################
        segmentation_annotations_list = []
        segmentation_annotations = dataset_dict['annotations']
        for index, item in enumerate(segmentation_annotations):
            item['bbox'] = np.array([item['bbox']])
            item['bbox'][:,2] = item['bbox'][:,0] + item['bbox'][:,2]
            item['bbox'][:,3] = item['bbox'][:, 1] + item['bbox'][:, 3]
            item['bbox_mode'] = BoxMode.XYXY_ABS
            item['segmentation'] = np.array(item['segmentation']).reshape(int(len(item['segmentation'][0])/2), 2)
            item['segmentation'] = item['segmentation'] / [1920, 1080]
            segmentation_annotations_list.append(item)
        segmentation_annotations = segmentation_annotations_list
        ##########################################################

        gaze_box = gaze_related_ann['gaze_box']
        gaze_box = np.array([gaze_box])
        gaze_box = gaze_box.astype(np.int32)
        # head_box = gaze_related_ann['head_box']
        # head_box = np.array([head_box])
        # head_box = head_box.astype(np.int32)

        eye = [float(gaze_related_ann['hx']) / 1920, float(gaze_related_ann['hy']) / 1080]
        gaze = [float(gaze_related_ann['gaze_cx']) / 1920, float(gaze_related_ann['gaze_cy']) / 1080]
        img = Image.open(dataset_dict['file_name'])
        img = img.convert('RGB')
        gaze_x, gaze_y = gaze
        eye_x, eye_y = eye

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

        if self.is_train:
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)
                Jitter = True
            # Random Crop
            if np.random.random_sample() <= 0.5:
                Crop = True
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                eye_x, eye_y = (eye_x * width - offset_x) / float(crop_width), \
                                 (eye_y * height - offset_y) / float(crop_height)
                segmentation_annotations_list = []
                for index, item in enumerate(segmentation_annotations):
                    item['bbox'][:, [0, 2]] = item['bbox'][:, [0, 2]]- crop_x_min
                    item['bbox'][:, [1, 3]] = item['bbox'][:, [1, 3]] - crop_y_min
                    item['segmentation'][:,0],  item['segmentation'][:,1] = (item['segmentation'][:,0] * width - offset_x) / float(crop_width), \
                                 (item['segmentation'][:,1] * height - offset_y) / float(crop_height)
                    segmentation_annotations_list.append(item)
                segmentation_annotations = segmentation_annotations_list

                width, height = crop_width, crop_height

                # head_box[:, [0, 2]] = head_box[:, [0, 2]] - crop_x_min
                # head_box[:, [1, 3]] = head_box[:, [1, 3]] - crop_y_min

                # operate gt_box
                gaze_box[:, [0, 2]] = gaze_box[:, [0, 2]] - crop_x_min
                gaze_box[:, [1, 3]] = gaze_box[:, [1, 3]] - crop_y_min



            # Random flip
            if np.random.random_sample() <= 0.5:
                flip = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x = 1 - eye_x
                segmentation_annotations_list = []
                for index, item in enumerate(segmentation_annotations):
                    item['bbox'][:, [0, 2]] = width - item['bbox'][:, [2, 0]]
                    item['segmentation'][:, 0] = 1 - item['segmentation'][:,0]
                    segmentation_annotations_list.append(item)
                segmentation_annotations = segmentation_annotations_list

                # head_box[:, [0, 2]] = width - head_box[:, [2, 0]]
                gaze_box[:, [0, 2]] = width - gaze_box[:, [2, 0]]



            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        # gaze_p_ = np.array([gaze_x, gaze_y])
        # eye_p_ = np.array([eye_x, eye_y])
        # gaze_direction_ = gaze_p_ - eye_p_

        gaze_p = np.array([gaze_x, gaze_y])
        eye_p = np.array([eye_x, eye_y])
        gaze_direction = gaze_p - eye_p
        gaze_direction_normal = np.linalg.norm(gaze_direction)
        if gaze_direction.mean() != 0:
            gaze_direction = gaze_direction / gaze_direction_normal
        gaze_direction = torch.FloatTensor(gaze_direction)

        # gaze_direction_ = gaze_direction.unsqueeze(0)
        # gaze_field = generate_gaze_field(eye_p)
        # gaze_field = torch.FloatTensor(gaze_field).unsqueeze(0)
        # # generate gaze field map
        # batch_size, channel, height, width = gaze_field.size()
        # gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        # gaze_field = gaze_field.view([batch_size, -1, 2])
        # gaze_field = torch.matmul(gaze_field, gaze_direction_.view([batch_size, 2, 1]))
        # gaze_cone = gaze_field.view([batch_size, height, width, 1])
        # gaze_cone = gaze_cone.permute([0, 3, 1, 2]).contiguous()
        # gaze_cone = nn.ReLU()(gaze_cone)
        # gaze_cone = gaze_cone.squeeze(0).squeeze(0)

        dataset_dict['gaze'] = [gaze_x, gaze_y]
        dataset_dict['eye'] = [eye_x, eye_y]
        dataset_dict['gaze_related_ann']['gaze_cx'] = gaze_x
        dataset_dict['gaze_related_ann']['gaze_cy'] = gaze_y
        dataset_dict['gaze_related_ann']['hx'] = eye_x
        dataset_dict['gaze_related_ann']['hy'] = eye_y
        dataset_dict['gaze_direction'] = gaze_direction

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height

        head_channel = utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                         resolution=224, coordconv=False).unsqueeze(0)
        dataset_dict['head_channel'] = head_channel
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face = face.resize((224, 224), Image.BICUBIC)
        face = np.array(face)
        face = torch.as_tensor(np.ascontiguousarray(face.transpose(2, 0, 1)))
        dataset_dict['face'] = face
        # face = self.transform(face)
        img = img.resize(([224, 224]), Image.BICUBIC)
        dataset_dict['ori_img'] = img

        im = img
        img = np.array(img)
        ims = img
        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
        dataset_dict['image'] = img
        # img = self.transform(img)

        # Bbox deal
        # head_box[:, [0, 2]] = head_box[:, [0, 2]] * 224 / width
        # head_box[:, [1, 3]] = head_box[:, [1, 3]] * 224 / height
        # hbox = head_box[0][0:4].tolist()

        # operate_gt_box
        gaze_box[:, [0, 2]] = gaze_box[:, [0, 2]] * 224 / width
        gaze_box[:, [1, 3]] = gaze_box[:, [1, 3]] * 224 / height

        gaze_box[:, 0:2][gaze_box[:, 0:2] < 0] = 0
        gaze_box[:, 2:4][gaze_box[:, 2:4] > 224] = 224

        head_box_large = np.array([[(eye_x-0.2)*224, (eye_y-0.2)*224, (eye_x+0.2)*224, (eye_y+0.2)*224, 1]], dtype=np.int32)

        # head_box_large = np.array([[x_min / width * 224, y_min / height * 224, x_max / width * 224, y_max / height * 224, 1]], dtype=np.int32)

        dataset_dict['gaze_related_ann']['gaze_box'] = gaze_box
        dataset_dict['gaze_related_ann']['head_box'] = head_box_large

        segmentation_annotations_list = []
        for index, item in enumerate(segmentation_annotations):
            box = item['bbox']
            box[:, [0, 2]] = box[:, [0, 2]] * 224 / width
            box[:, [1, 3]] = box[:, [1, 3]] * 224 / height
            item['bbox'] = box.astype(np.float64)

            seg = item['segmentation']
            seg[:,0] = seg[:,0] * 224
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

            if box.shape[0] !=0:
                box = box[0]
                item['bbox'] = box
                new_segmentation_annotations.append(item)
        dataset_dict['annotations'] = new_segmentation_annotations
        gaze_heatmap = torch.zeros(64, 64)  # set the size of the output
        gaze_heatmap = utils.draw_labelmap(gaze_heatmap, [gaze_x * 64, gaze_y * 64],
                                                   3,
                                                   type='Gaussian')
        dataset_dict['gaze_heatmap'] = gaze_heatmap

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)
            annos = new_segmentation_annotations

            ################################################
            # generate gaze item mask
            ################################################
            gaze_item_ = []
            max_iou = 0
            for item_ann in annos:
                item_box = item_ann['bbox'].astype(np.int32)
                gaze_box_ = gaze_box[0][0:4]
                iou = compute_iou(item_box, gaze_box_)
                if iou > max_iou:
                    gaze_item_ = item_ann
                max_iou = iou
            # gaze_item_category = gaze_item_['category_id']
            gaze_item_mask = torch.zeros([64,64], dtype=torch.uint8)
            if gaze_item_ == []:
                gaze_item_mask = gaze_item_mask
            else:
                gaze_item_segmentation = [[gaze_item_['segmentation'][0] / 224 * 64]]
                gaze_item_mask = convert_coco_poly_to_mask(gaze_item_segmentation, 64, 64)
                gaze_item_mask = gaze_item_mask.squeeze(0)
            if torch.sum(gaze_item_mask).item() == 0:
                gaze_x_index = int(gaze_x * 64)
                gaze_y_index = int(gaze_y * 64)
                gaze_item_mask[(gaze_x_index - 1) : (gaze_x_index + 2), (gaze_y_index - 1) : (gaze_y_index + 2)] = 1
            dataset_dict['gaze_related_ann']['gaze_item_mask'] = gaze_item_mask
            ###################################################
            # x=0

            #####################################
            # visualization check
            # draw = ImageDraw.Draw(im)
            # for item in annos:
            #     bbox = item['bbox'].astype(int)
            #     bbox = bbox.tolist()
            #     draw.rectangle(bbox, fill=None, outline='green', width=3)
            # draw.rectangle(gaze_box[0][:4].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle(head_box_large[0].tolist(), fill=None, outline='red', width=3)
            # # draw.rectangle([eye_x * 224 - 40, eye_y * 224 - 40, eye_x * 224 + 40, eye_y * 224 + 40],fill=None, outline='red', width=3)
            # draw.ellipse([gaze_x*224 - 2, gaze_y*224 - 2, gaze_x*224 + 2, gaze_y*224 + 2],fill='red', outline=None)
            # draw.ellipse([eye_x*224- 2, eye_y*224 - 2, eye_x*224 + 2, eye_y*224 + 2], fill='red',outline=None)
            # draw.line([eye_x*224, eye_y*224, gaze_x*224, gaze_y*224],fill='yellow',width=2)
            # # im_with_hm = utils.generate_attention_map(im, gaze_cone)
            # # im_with_hm.show()
            # im.show()
            # x=0

            instances = utils.annotations_to_instances_goo(annos, (224, 224))

            if not instances.has('gt_masks'):  # this is to avoid empty annotation
                instances.gt_masks = PolygonMasks([])
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

            # import cv2
            # re = torch.sum(gt_masks, dim=0)
            # re = re.numpy()
            # overlay = np.ones_like(ims) * 255
            # overlay[re == 0] = 0
            # result = cv2.addWeighted(ims, 0.5, overlay, 0.5, 0.2)
            # cv2.imwrite('image.jpg', result)

        return dataset_dict

def generate_gaze_field(head_position):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 7, 7
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = head_position
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid