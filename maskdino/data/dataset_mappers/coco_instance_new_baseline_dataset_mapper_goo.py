# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging
from PIL import Image
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, PolygonMasks,BoxMode
import torchvision.transforms.functional as TF
from pycocotools import mask as coco_mask

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution - 1)
    if coordconv:
        unit = np.array(range(0, resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit + i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3], head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution, resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3], head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img / np.max(img)  # normalize heatmap so it has max value of 1
    return to_torch(img)


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
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
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
        from PIL import Image,ImageDraw
        import cv2
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = Image.open(dataset_dict["file_name"])
        image = image.resize(([640, 480]),Image.BICUBIC)
        image_size = image.size
        width, height = image_size

        # padding_mask = np.ones([height, width])
        # im = Image.fromarray(image)
        # im.show()
        #########################################################
        # head_box = dataset_dict['gaze_related_ann']['head_box']
        # head_image = image[head_box[1]:head_box[3],head_box[0]:head_box[2]]
        # head_image = cv2.resize(head_image, (1024, 1024))
        # head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('image.jpg', head_image)
        #########################################################
        annotations = dataset_dict['annotations']
        new_annotaions_list = []
        for index, item in enumerate(annotations):
            item['bbox'] = np.array([item['bbox']])/[1920 , 1080, 1920, 1080]*[640, 480, 640, 480]
            item['bbox'] = item['bbox'].astype(np.int32)
            item['bbox'][:, 2] = item['bbox'][:, 0] + item['bbox'][:, 2]
            item['bbox'][:, 3] = item['bbox'][:, 1] + item['bbox'][:, 3]
            item["bbox_mode"] = BoxMode.XYXY_ABS
            item['segmentation'] = np.array(item['segmentation'][0]).astype(np.int32)
            item['segmentation'] = item['segmentation'].reshape(int(item['segmentation'].shape[0]/2), 2)
            item['segmentation'] = item['segmentation']/[1920, 1080]*[640, 480]
            new_annotaions_list.append(item)
            x=0
        # dataset_dict['annotations'] = new_annotaions_list

        gaze_annotations = dataset_dict['gaze_related_ann']
        gaze_x = gaze_annotations['gaze_cx']/640
        gaze_y = gaze_annotations['gaze_cy']/480
        eye_x = gaze_annotations['hx']/640
        eye_y = gaze_annotations['hy']/480
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
            # data augmentation
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)
            # Random Crop
            if np.random.random_sample() <= 0.5:
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
                image = TF.crop(image, crop_y_min, crop_x_min, crop_height, crop_width)
                # padding_mask = padding_mask[crop_x_min: crop_x_min+crop_width, crop_y_min: crop_y_min+crop_height]

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                eye_x, eye_y = (eye_x * width - offset_x) / float(crop_width), \
                                 (eye_y * height - offset_y) / float(crop_height)
                new_annotaions_list_crop = []
                for i, item in enumerate(new_annotaions_list):
                    item['bbox'][:, [0, 2]] = item['bbox'][:, [0, 2]] - crop_x_min
                    item['bbox'][:, [1, 3]] = item['bbox'][:, [1, 3]] - crop_y_min
                    item['segmentation'][:,0], item['segmentation'][:,1] = (item['segmentation'][:,0] - offset_x), \
                                                                           (item['segmentation'][:,1] - offset_y)
                    new_annotaions_list_crop.append(item)
                new_annotaions_list = new_annotaions_list_crop
                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # padding_mask = np.flip(padding_mask, axis=1)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x = 1 - eye_x
                new_annotaions_list_flip = []
                for i, item in enumerate(new_annotaions_list):
                    item['bbox'][:, [0, 2]] = width - item['bbox'][:, [2, 0]]
                    item['segmentation'][:,0] = width - item['segmentation'][:,0]
                    new_annotaions_list_flip.append(item)
                new_annotaions_list = new_annotaions_list_flip

            # Random color change
            if np.random.random_sample() <= 0.5:
                image = TF.adjust_brightness(image, brightness_factor=np.random.uniform(0.5, 1.5))
                image = TF.adjust_contrast(image, contrast_factor=np.random.uniform(0.5, 1.5))
                image = TF.adjust_saturation(image, saturation_factor=np.random.uniform(0, 1.5))

            # Random color change
            if np.random.random_sample() <= 0.5:
                image = TF.adjust_brightness(image, brightness_factor=np.random.uniform(0.5, 1.5))
                image = TF.adjust_contrast(image, contrast_factor=np.random.uniform(0.5, 1.5))
                image = TF.adjust_saturation(image, saturation_factor=np.random.uniform(0, 1.5))

        new_annotaions_list_final = []
        for i, item in enumerate(new_annotaions_list):
            item['segmentation'][:, 0] = item['segmentation'][:, 0] * 224 / width
            item['segmentation'][:, 1] = item['segmentation'][:, 1] * 224 / height
            item['segmentation'][:, 0][item['segmentation'][:, 0] < 0] = 0
            item['segmentation'][:, 1][item['segmentation'][:, 1] > 224] = 224
            item['segmentation'] = [item['segmentation'].astype(np.int32).flatten()]
            # arr = item['segmentation']
            # item['segmentation'] = np.unique(arr, axis=0)

            item['bbox'][:, [0, 2]] = item['bbox'][:, [0, 2]] * 224 / width
            item['bbox'][:, [1, 3]] = item['bbox'][:, [1, 3]] * 224 / height
            item['bbox'][:, 0:2][item['bbox'][:, 0:2] < 0] = 0
            item['bbox'][:, 2][item['bbox'][:, 2] > 224] = 224
            item['bbox'][:, 3][item['bbox'][:, 3] > 224] = 224
            box_w = item['bbox'][:, 2] - item['bbox'][:, 0]
            box_h = item['bbox'][:, 3] - item['bbox'][:, 1]
            item['bbox'] = item['bbox'][np.logical_and(box_w > 1, box_h > 1)]
            if item['bbox'].shape[0] != 0:
                item['bbox'] = item['bbox'][0]
                new_annotaions_list_final.append(item)

        x=0
        face = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face = face.resize(([224, 224]), Image.BICUBIC)
        face = np.array(face)
        image = image.resize(([224, 224]), Image.BICUBIC)
        # im = image
        image = np.array(image)
        # padding_mask = cv2.resize(padding_mask,)
        # hbox = [x_min, y_min, x_max, y_max]
        # #before transform
        # head_image = image[y_min:y_max, x_min:x_max]
        # head_image = cv2.resize(head_image, (224,224))
        # head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
        #

        # cv2.imwrite('image.jpg', head_image)
        # utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms


        # image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # image_shape = image.shape[:2]
        # image = cv2.resize(image, (224, 224))
        # im = Image.fromarray(image)
        # im.show()
        # the crop transformation has default padding value 0 for segmentation
        # padding_mask = transforms.apply_segmentation(padding_mask)
        # padding_mask = ~ padding_mask.astype(bool)

          # h, w

        # hbox = utils.transform_box_annotations(hbox,transforms, image_shape)
        # x_min, y_min, x_max, y_max = hbox.tolist()
        # head_channel = get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
        #                                                  resolution=224, coordconv=False).unsqueeze(0)
        # # gaze_related_ann = dataset_dict['gaze_related_ann']
        # gaze_point = [gaze_x*1920, gaze_y*1080]
        # eye_point = [eye_x*1920, eye_y*1080]
        # points = [gaze_x*1920, gaze_y*1080, eye_x*1920, eye_y*1080]
        # points = utils.transform_keypoint_annotations(points, transforms, image_shape)
        # gaze_point = points[0].tolist()/[1920, 1080]*[224,224]
        # eye_point = points[1].tolist()/[1920, 1080]*[224,224]

        # x=0

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["face"] = torch.as_tensor(np.ascontiguousarray(face.transpose(2, 0, 1)))
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        dataset_dict['annotations'] = new_annotaions_list_final
        dataset_dict['gaze_cx'] = gaze_x
        dataset_dict['gaze_cy'] = gaze_y
        dataset_dict['hx'] = eye_x
        dataset_dict['hy'] = eye_y

        gaze_heatmap = torch.zeros(64, 64)  # set the size of the output
        gaze_heatmap = draw_labelmap(gaze_heatmap, [gaze_x * 64, gaze_y * 64],
                                                  3,
                                                  type='Gaussian')

        # dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        # dataset_dict["head_image"] = torch.as_tensor(np.ascontiguousarray(head_image.transpose(2, 0, 1)))
        # dataset_dict["head_channel"] = torch.as_tensor(head_channel)
        # dataset_dict['gaze'] = gaze_point
        # dataset_dict['eye'] = eye_point
        dataset_dict['gaze_related_ann']['gaze_cx'] = gaze_x
        dataset_dict['gaze_related_ann']['gaze_cy'] = gaze_y
        dataset_dict['gaze_related_ann']['hx'] = eye_x
        dataset_dict['gaze_related_ann']['hy'] = eye_y
        dataset_dict['gaze_related_ann']['heatmap'] = gaze_heatmap


        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            annos = dataset_dict['annotations']
            #####################################
            # visualization check
            # draw = ImageDraw.Draw(im)
            # for item in annos:
            #     bbox = item['bbox'].astype(int)
            #     bbox = bbox.tolist()
            #     draw.rectangle(bbox, fill=None, outline='green', width=3)
            # # draw.rectangle(hbox.tolist(), fill=None, outline='red', width=3)
            # draw.ellipse([gaze_x*224 - 2, gaze_y*224 - 2, gaze_x*224 + 2, gaze_y*224 + 2],fill='red', outline=None)
            # draw.ellipse([eye_x*224 - 2, eye_y*224 - 2, eye_x*224 + 2, eye_y*224 + 2], fill='red',outline=None)
            # draw.line([eye_x, eye_y, gaze_x, gaze_y],fill='yellow',width=2)
            # im.show()
            #####################################
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, (224,224))
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
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
            # re = torch.sum(gt_masks, dim=0)
            # re = re.numpy()
            # import cv2
            # overlay = np.ones_like(image) * 255
            # overlay[re == 0] = 0
            # result = cv2.addWeighted(image, 0.5, overlay, 0.5, 0.2)
            # cv2.imwrite('image.jpg', result)
            dataset_dict["instances"] = instances

        return dataset_dict
