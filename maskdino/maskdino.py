# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .transformer import Transformer
import math

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils import box_ops
from collections import OrderedDict
from .resnet_bk import resnet50, resnet101, resnet152
import numpy as np
from torchvision.ops import RoIAlign

# torchvision.ops.RoIAlign(output_size, spatial_scale, sampling_ratio)



class Resnet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(Resnet, self).__init__()
        self.edition = [resnet50, resnet101, resnet152]
        self.model = resnet50(pretrained)

    def forward(self, image, face):

        image = self.model.layer5_scene(image)
        face = self.model.layer5_face(face)

        return image, face

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class Pixel_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel_shuffle, self).__init__()

        self.pixel_shuffle = nn.Sequential(
            nn.PixelShuffle(2),
            conv2d(in_channels, out_channels, 3)
        )

    def forward(self, x, ):
        x = self.pixel_shuffle(x)
        return x

class Pixel_shuffle1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel_shuffle1, self).__init__()

        self.pixel_shuffle = nn.Sequential(
            nn.PixelShuffle(4),
            conv2d(in_channels, out_channels, 3)
        )

    def forward(self, x, ):
        x = self.pixel_shuffle(x)
        return x

def select_head(outputs, target_sizes):
    num_select = 300
    out_logits, out_bbox = outputs['pred_head_logits'], outputs['pred_head_boxes']

    # assert len(out_logits) == len(target_sizes)
    # assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]


    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    # img_h, img_w = target_sizes.unbind(1)
    # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * target_sizes

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    heads = []
    for output in results:
        scores_ = output['scores']
        labels_ = output['labels']
        boxes_ = output['boxes']

        head_label_id = labels_ == 1
        labels = labels_[head_label_id]
        scores = scores_[head_label_id]
        boxes = boxes_[head_label_id]

        if len(labels) > 1:
            max = torch.max(scores)
            mask = scores == max
            labels = labels[mask]
            scores = scores[mask]
            boxes = boxes[mask]
            if labels.shape[0] > 1:
                labels = labels[0].unsqueeze(0)
                scores = scores[0].unsqueeze(0)
                boxes = boxes[0].unsqueeze(0)
        else:
            max = torch.max(scores_)
            mask = scores_ == max
            labels = labels_[mask]
            scores = scores_[mask]
            boxes = boxes_[mask]
            if labels.shape[0] > 1:
                labels = labels[0].unsqueeze(0)
                scores = scores[0].unsqueeze(0)
                boxes = boxes[0].unsqueeze(0)

        dict = {'scores': scores, 'labels': labels, 'boxes': boxes}
        heads.append(dict)
    return heads

def get_head_channel(pred_head, gt_heads, train_=False):

    head_channels = []
    is_head = []
    for i, boxes in enumerate(pred_head):
        box = boxes['boxes']
        conf = boxes['scores']
        if not train_:
            if conf.shape[0] != 0 and conf[0].item() > 0.5:
                box1 = box[0]
                is_head.append(1)
                head_c = torch.zeros(1, 224, 224)
                xmin, ymin, xmax, ymax = box1
                # xmin = xmin * 224
                # ymin = ymin * 224
                # xmax = xmax * 224
                # ymax = ymax * 224
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > 224:
                    xmax = 224
                if ymax > 224:
                    ymax = 224

                head_c[:, ymin:ymax + 1, xmin:xmax + 1] = 1
                head_channels.append(head_c)
            else:
                head_c = torch.zeros(1, 224, 224)
                is_head.append(0)
                head_channels.append(head_c)
        else:
            if box.numel() == 0:
                head_c = torch.zeros(1, 224, 224)
                is_head.append(0)
                head_channels.append(head_c)
            else:
                box1 = box[0]
                box2 = gt_heads[i]
                box1 = box1.to(box2.dtype)
                # 计算相交区域的左上角和右下角坐标
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                # 计算相交区域面积
                intersection = max(0, x2 - x1) * max(0, y2 - y1)

                # 计算两个框各自面积
                area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # 计算并集面积
                union = area_box1 + area_box2 - intersection

                # 计算 IOU
                iou = intersection / union
                if iou > 0.5:
                    is_head.append(1)
                    head_c = torch.zeros(1, 224, 224)
                    xmin, ymin, xmax, ymax = box1
                    # xmin = xmin
                    # ymin = ymin
                    # xmax = xmax
                    # ymax = ymax
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)

                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax > 224:
                        xmax = 224
                    if ymax > 224:
                        ymax = 224

                    head_c[:, ymin:ymax + 1, xmin:xmax + 1] = 1
                    head_channels.append(head_c)
                else:
                    head_c = torch.zeros(1, 224, 224)
                    is_head.append(0)
                    head_channels.append(head_c)

    head_channels = torch.cat([items.unsqueeze(0) for items in head_channels], 0)

    head_channels_ = head_channels.cuda()

    return head_channels_, is_head


@META_ARCH_REGISTRY.register()
class MaskDINO(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        # #direction estimation
        # self.direction_fc = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 2),
        #     nn.Tanh())

        # GOO network
        # common
        self.head_proposal_align = RoIAlign(output_size=(7, 7), spatial_scale= 7 / 224, sampling_ratio=2, aligned=True)
        self.resnet_block = Resnet(phi='resnet50', pretrained=True)
        self.relu = nn.ReLU(inplace=True)

        # pxiel shuffle
        self.ps1 = Pixel_shuffle1(16, 256)
        # self.ps1 = Pixel_shuffle(64, 256)
        self.ps2 = Pixel_shuffle(128, 512)
        self.ps3 = Pixel_shuffle(256, 1024)
        self.ps4 = Pixel_shuffle(512, 2048)

        self.head_conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU()
        )

        self.mask_feat_conv_block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # head pathway
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool_roi = nn.AvgPool2d(3, stride=1)
        # direction estimation
        self.direction_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh())

        self.conv_face_scene = nn.Conv2d(2560, 2048, kernel_size=1)

        # attention
        self.attn = nn.Linear(1808, 1 * 7 * 7)

        self.relu = nn.ReLU(inplace=True)
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)

        self.totrans_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.totrans_conv_bn1 = nn.BatchNorm2d(256)

        self.query_embed = nn.Embedding(225, 256)
        self.transformer_layer = Transformer(d_model=256, dropout=0.0,
                                             nhead=8, dim_feedforward=2048,
                                             num_encoder_layers=1, num_decoder_layers=1,
                                             normalize_before=False, return_intermediate_dec=True,
                                             )
        self.conv_trblock = nn.Conv2d(256, 256, kernel_size=1)
        self.trblock_bn = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)


        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print('criterion.weight_dict ', self.criterion.weight_dict)



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        gaze_weight = 1000
        gaze_direction_weight = 10
        gaze_mask_energy_weight = 1
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})


        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)

            # head_interm_weight_dict = {}
            # head_interm_weight_dict.update({k + f'_interm_head': v for k, v in weight_dict.items()})
            # weight_dict.update(head_interm_weight_dict)

        weight_dict.update({"loss_gaze": gaze_weight})
        weight_dict.update({"loss_direction": gaze_direction_weight})
        weight_dict.update({"loss_mask_energy": gaze_mask_energy_weight})

        weight_dict.update({"loss_ce_head": class_weight})
        weight_dict.update({"loss_bbox_head": box_weight, "loss_giou_head": giou_weight})

        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]
        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        head_boxes = None
        if self.training:
            head_targets = []
            head_boxes = []
            gt_gaze_heatmap = [x["gaze_heatmap"].to(self.device) for x in batched_inputs]
            for i, item in enumerate(batched_inputs):
                dict_ = dict()
                head_box = item['gaze_related_ann']['head_box'][:, :4][0]
                head_box_ = head_box
                head_box = torch.as_tensor(head_box).cuda()
                head_box_ = torch.as_tensor(head_box_).cuda()
                head_target_box = head_box_.unsqueeze(0)
                size_ = torch.tensor([224., 224., 224., 224.]).cuda()
                head_target_box = box_ops.box_xyxy_to_cxcywh(head_target_box) / size_
                head_target_labels = torch.tensor([1.]).cuda()
                dict_['labels'] = head_target_labels
                dict_['boxes'] = head_target_box
                head_targets.append(dict_)
                head_boxes.append(head_box)
            # x=0
            gt_gaze_direction = [x["gaze_direction"].to(self.device) for x in batched_inputs]
            gt_gaze_item_mask = [x["gaze_related_ann"]['gaze_item_mask'].to(self.device) for x in batched_inputs]

        # eye_position = [x["eye"] for x in batched_inputs]
        # gaze_filed_list = []
        # for eye_p in eye_position:
        #     gaze_field_ = generate_gaze_field(eye_p)
        #     gaze_field_ = torch.FloatTensor(gaze_field_)
        #     gaze_filed_list.append(gaze_field_)
        # gaze_field = torch.stack(gaze_filed_list).to(self.device)
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # faces = [x["face"].to(self.device) for x in batched_inputs]
        # faces = [(x - self.pixel_mean) / self.pixel_std for x in faces]
        # faces = ImageList.from_tensors(faces, self.size_divisibility)

        # heads_channel = [x["head_channel"].to(self.device) for x in batched_inputs]
        # heads_channel = torch.stack(heads_channel)


        features = self.backbone(images.tensor)
        # face_feature = self.backbone(images.tensor)
        # face_feature = face_feature['res5']
        scene_feature = features['res5']
        # scene_feature_detach = scene_feature_undetach.clone().detach()

        features['res2'] = self.ps1(features['res2'])
        features['res3'] = self.ps2(features['res3'])
        features['res4'] = self.ps3(features['res4'])
        features['res5'] = self.ps4(features['res5'])

        # new_feature = {}
        # index = 2
        # for value in features.values():
        #     feat = F.interpolate(value, size=(value.shape[2] * 4, value.shape[3] * 4), mode='bilinear',
        #                                 align_corners=False)
        #     new_feature['res' + str(index)] = feat
        #     index += 1
        # features = new_feature



        # head_embeding = face_feature.clone()
        # head_embeding = self.avgpool(head_embeding).view(-1, 2048)
        # gaze_direction = self.direction_fc(head_embeding)
        # normalized_direction = gaze_direction / gaze_direction.norm(dim=1).unsqueeze(1)

        # generate gaze field map
        # batch_size, channel, height, width = gaze_field.size()
        # gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        # gaze_field = gaze_field.view([batch_size, -1, 2])
        # gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, 2, 1]))
        # gaze_cone = gaze_field.view([batch_size, height, width, 1])
        # gaze_cone = gaze_cone.permute([0, 3, 1, 2]).contiguous()
        # gaze_cone = nn.ReLU()(gaze_cone)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if 'detr' in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images)
                else:
                    targets = self.prepare_targets(gt_instances, gt_gaze_heatmap, gt_gaze_direction, images)
            else:
                targets = None
            outputs, mask_dict, object_mem, object_pos, object_mask, head_outputs = self.sem_seg_head(features, targets=targets)

            #####################################################
            # Get Mask Tensor
            #####################################################
            # pred_mask = outputs['pred_masks']
            # mask_cls_results = outputs['pred_logits']
            # mask_pred_results = F.interpolate(
            #     pred_mask,
            #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            # mask_tensor_list = []
            # for mask_pred_results_, mask_cls_result_ in zip(mask_pred_results, mask_cls_results):
            #     width = 224
            #     height = 224
            #     mask_pred_results_ = retry_if_cuda_oom(sem_seg_postprocess)(
            #         mask_pred_results_, [224, 224], height, width
            #     )
            #
            #     instance_r = self.instance_training(mask_cls_result_, mask_pred_results_)
            #     mask = instance_r.pred_masks
            #     mask_tensor_list.append(mask)
            # mask_tensor = torch.stack(mask_tensor_list).to(self.device)
            #####################################################

            orig_target_sizes = 224

            heads = select_head(head_outputs, orig_target_sizes)
            roi_list = []
            pred_eye_point_list = []
            for i, item in enumerate(heads):
                batch_idx = torch.tensor([[float(i)]]).cuda()
                head_box_for_roi = item['boxes']
                eye = [(head_box_for_roi[:,0]+head_box_for_roi[:,2]).item() / 2, (head_box_for_roi[:,1]+head_box_for_roi[:,3]).item() / 2]
                eye = [eye[0] / orig_target_sizes, eye[1] / orig_target_sizes]
                pred_eye_point_list.append(eye)
                roi_box = torch.cat([batch_idx.expand(head_box_for_roi.size(0), -1), head_box_for_roi], dim=1)
                roi_list.append(roi_box)
            head_proposual = torch.cat([items for items in roi_list], 0)
            roi_align_out = self.head_proposal_align(scene_feature, head_proposual)

            scene_feat, face_feat = self.resnet_block(scene_feature, roi_align_out)

            roi_head_flatten = self.avgpool(roi_align_out).view(-1, 2048)
            gaze_direction = self.direction_fc(roi_head_flatten)
            normalized_direction = gaze_direction / gaze_direction.norm(dim=1).unsqueeze(1)

            eye_position = pred_eye_point_list
            gaze_filed_list = []
            for eye_p in eye_position:
                gaze_field_ = generate_gaze_field(eye_p)
                gaze_field_ = torch.FloatTensor(gaze_field_)
                gaze_filed_list.append(gaze_field_)
            gaze_field = torch.stack(gaze_filed_list).to(self.device)

            # generate gaze field map
            batch_size, channel, height, width = gaze_field.size()
            gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
            gaze_field = gaze_field.view([batch_size, -1, 2])
            gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, 2, 1]))
            gaze_cone = gaze_field.view([batch_size, height, width, 1])
            gaze_cone = gaze_cone.permute([0, 3, 1, 2]).contiguous()
            gaze_cone = nn.ReLU()(gaze_cone)

            gaze_cone = torch.pow(gaze_cone, 2)
            # m=0
            ########################################################
            # predicted head box visualization
            # from PIL import Image,ImageDraw
            # for batch, head in zip(batched_inputs, heads):
            #     head_gt_ = batch['gaze_related_ann']['head_box'][:,0:4]
            #     im = batch['ori_img']
            #     head_box_vis = head['boxes']
            #     head_box_vis[0][0] = head_box_vis[0][0]
            #     head_box_vis[0][1] = head_box_vis[0][1]
            #     head_box_vis[0][2] = head_box_vis[0][2]
            #     head_box_vis[0][3] = head_box_vis[0][3]
            #     head_box_vis = head_box_vis.detach().cpu().numpy()
            #     # head_box_vis[:,[0, 1]] = head_box_vis[:,[1, 0]]
            #     # head_box_vis[:,[2, 3]] = head_box_vis[:,[3, 2]]
            #
            #     draw = ImageDraw.Draw(im)
            #     draw.rectangle(head_box_vis[0].tolist(), fill=None, outline='red', width=3)
            #     draw.rectangle(head_gt_[0].tolist(), fill=None, outline='green', width=3)
            #     im.show()
            #     z=0
            ########################################################
            if not self.training:
                head_boxes = []
            heads_channel, is_head = get_head_channel(heads, head_boxes, self.training)
            # GOO
            # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
            head_reduced = self.maxpool(self.maxpool(self.maxpool(heads_channel))).view(-1, 784)
            # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
            # get and reshape attention weights such that it can be multiplied with scene feature map
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            attn_weights = attn_weights.view(-1, 1, 49)
            attn_weights = F.softmax(attn_weights, dim=2)  # soft attention weights single-channel
            attn_weights = attn_weights.view(-1, 1, 7, 7)
            attn_weights = torch.mul(gaze_cone, attn_weights)

            head_conv = self.head_conv_block(heads_channel)
            # attn_weights = torch.ones(attn_weights.shape)/49.0
            scene_feat = torch.cat((scene_feat, head_conv), 1)
            attn_applied_scene_feat = torch.mul(attn_weights,
                                                scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

            scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
            scene_face_feat = self.conv_face_scene(scene_face_feat)

            # scene + face feat -> encoding -> decoding
            encoding = self.compress_conv1(scene_face_feat)
            encoding = self.compress_bn1(encoding)
            encoding = self.relu(encoding)
            encoding = self.compress_conv2(encoding)
            encoding = self.compress_bn2(encoding)
            encoding = self.relu(encoding)
            x_gaze = self.deconv1(encoding)
            x_gaze = self.deconv_bn1(x_gaze)
            x_gaze = self.relu(x_gaze)

            src_ = self.totrans_conv(x_gaze)
            src_ = self.totrans_conv_bn1(src_)
            src_ = self.relu(src_)

            # device = tensor_list.tensors.device
            bs_, c_, h_, w_ = x_gaze.shape
            mask_shape_ = (bs_, h_, w_)  # 3 dimensions with size 4, 5 and 6 respectively
            mask_ = torch.zeros(mask_shape_, dtype=torch.bool).to('cuda')
            # mask = tensor_list.mask
            pos_ = make_pos(mask_, c_ / 2)

            hs_ = self.transformer_layer(src_, mask_, self.query_embed.weight, pos_, object_mem, object_pos, object_mask)[0]
            hs_ = hs_.reshape(bs_, h_, w_, c_).permute(0, 3, 1, 2)
            # scene + face feat -> encoding -> decoding
            cross_features = self.conv_trblock(hs_)
            cross_features = self.trblock_bn(cross_features)
            cross_features = self.relu(cross_features)

            x_gaze = self.deconv2(cross_features)
            x_gaze = self.deconv_bn2(x_gaze)
            x_gaze = self.relu(x_gaze)
            x_gaze = self.deconv3(x_gaze)
            x_gaze = self.deconv_bn3(x_gaze)
            x_gaze = self.relu(x_gaze)
            x_gaze = self.conv4(x_gaze)

            # normalized_direction = 0

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, x_gaze, normalized_direction, head_targets, gt_gaze_item_mask, mask_dict)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                # else:
                #     # remove this loss if not specified in `weight_dict`
                #     losses.pop(k)
            return losses
        else:
            outputs, _, object_mem, object_pos, object_mask, head_outputs = self.sem_seg_head(features)

            orig_target_sizes = 224

            heads = select_head(head_outputs, orig_target_sizes)

            roi_list = []
            pred_eye_point_list = []
            for i, item in enumerate(heads):
                batch_idx = torch.tensor([[float(i)]]).cuda()
                head_box_for_roi = item['boxes']
                eye = [(head_box_for_roi[:, 0] + head_box_for_roi[:, 2]).item() / 2,
                       (head_box_for_roi[:, 1] + head_box_for_roi[:, 3]).item() / 2]
                eye = [eye[0] / orig_target_sizes, eye[1] / orig_target_sizes]
                pred_eye_point_list.append(eye)
                roi_box = torch.cat([batch_idx.expand(head_box_for_roi.size(0), -1), head_box_for_roi], dim=1)
                roi_list.append(roi_box)
            head_proposual = torch.cat([items for items in roi_list], 0)
            roi_align_out = self.head_proposal_align(scene_feature, head_proposual)

            scene_feat, face_feat = self.resnet_block(scene_feature, roi_align_out)

            roi_head_flatten = self.avgpool(roi_align_out).view(-1, 2048)
            gaze_direction = self.direction_fc(roi_head_flatten)
            normalized_direction = gaze_direction / gaze_direction.norm(dim=1).unsqueeze(1)

            eye_position = pred_eye_point_list
            gaze_filed_list = []
            for eye_p in eye_position:
                gaze_field_ = generate_gaze_field(eye_p)
                gaze_field_ = torch.FloatTensor(gaze_field_)
                gaze_filed_list.append(gaze_field_)
            gaze_field = torch.stack(gaze_filed_list).to(self.device)

            # generate gaze field map
            batch_size, channel, height, width = gaze_field.size()
            gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
            gaze_field = gaze_field.view([batch_size, -1, 2])
            gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, 2, 1]))
            gaze_cone = gaze_field.view([batch_size, height, width, 1])
            gaze_cone = gaze_cone.permute([0, 3, 1, 2]).contiguous()
            gaze_cone = nn.ReLU()(gaze_cone)

            if head_boxes is None:
                head_boxes = []
            heads_channel, is_head = get_head_channel(heads, head_boxes)

            # GOO
            # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
            head_reduced = self.maxpool(self.maxpool(self.maxpool(heads_channel))).view(-1, 784)
            # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
            # get and reshape attention weights such that it can be multiplied with scene feature map
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            # attn_weights = attn_weights.view(-1, 1, 49)
            # attn_weights = attn_weights.view(-1, 1, 7, 7)
            # attn_weights = torch.mul(gaze_cone, attn_weights)
            # attn_weight = attn_weights
            attn_weights = attn_weights.view(-1, 1, 49)
            attn_weights = F.softmax(attn_weights, dim=2)
            # fusion = attn_weights# soft attention weights single-channel
            attn_weights = attn_weights.view(-1, 1, 7, 7)
            attn_weights = torch.mul(gaze_cone, attn_weights)

            head_conv = self.head_conv_block(heads_channel)
            # attn_weights = torch.ones(attn_weights.shape)/49.0
            scene_feat = torch.cat((scene_feat, head_conv), 1)
            attn_applied_scene_feat = torch.mul(attn_weights,
                                                scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

            scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
            scene_face_feat = self.conv_face_scene(scene_face_feat)

            # scene + face feat -> encoding -> decoding
            encoding = self.compress_conv1(scene_face_feat)
            encoding = self.compress_bn1(encoding)
            encoding = self.relu(encoding)
            encoding = self.compress_conv2(encoding)
            encoding = self.compress_bn2(encoding)
            encoding = self.relu(encoding)
            x_gaze = self.deconv1(encoding)
            x_gaze = self.deconv_bn1(x_gaze)
            x_gaze = self.relu(x_gaze)

            src_ = self.totrans_conv(x_gaze)
            src_ = self.totrans_conv_bn1(src_)
            src_ = self.relu(src_)

            # device = tensor_list.tensors.device
            bs_, c_, h_, w_ = x_gaze.shape
            mask_shape_ = (bs_, h_, w_)  # 3 dimensions with size 4, 5 and 6 respectively
            mask_ = torch.zeros(mask_shape_, dtype=torch.bool).to('cuda')
            # mask = tensor_list.mask
            pos_ = make_pos(mask_, c_ / 2)

            hs_ = self.transformer_layer(src_, mask_, self.query_embed.weight, pos_, object_mem, object_pos, object_mask)[0]
            hs_ = hs_.reshape(bs_, h_, w_, c_).permute(0, 3, 1, 2)
            # scene + face feat -> encoding -> decoding
            cross_features = self.conv_trblock(hs_)
            cross_features = self.trblock_bn(cross_features)
            cross_features = self.relu(cross_features)

            x_gaze = self.deconv2(cross_features)
            x_gaze = self.deconv_bn2(x_gaze)
            x_gaze = self.relu(x_gaze)
            x_gaze = self.deconv3(x_gaze)
            x_gaze = self.deconv_bn3(x_gaze)
            x_gaze = self.relu(x_gaze)
            x_gaze = self.conv4(x_gaze)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # mask_box_result = mask_box_result.to(mask_pred_result)
                    # mask_box_result = self.box_postprocess(mask_box_result, height, width)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["instances"] = instance_r
            # normalized_direction = torch.zeros([2,2]).cuda()
            return processed_results, x_gaze, normalized_direction,heads
            # return processed_results, attn_weights, normalized_direction, heads

    def prepare_targets(self, targets, gt_gaze_heamap, gt_direction, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, heatmap, dr in zip(targets, gt_gaze_heamap, gt_direction):
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            # z = targets_per_image.gt_boxes.tensor
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy,
                    "gaze_heatmap": heatmap.unsqueeze(0),
                    "gt_direction": dr
                }
            )
        return new_targets

    def prepare_targets_detr(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def instance_training(self,mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()

        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

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

def make_pos(mask, hidden_dim):

    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    scale = 2 * math.pi

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_tx = torch.arange(hidden_dim, dtype=torch.float32, device=mask.device)
    dim_tx = 20 ** (2 * (dim_tx // 2) / hidden_dim)
    pos_x = x_embed[:, :, :, None] / dim_tx

    dim_ty = torch.arange(hidden_dim, dtype=torch.float32, device=mask.device)
    dim_ty = 20 ** (2 * (dim_ty // 2) / hidden_dim)
    pos_y = y_embed[:, :, :, None] / dim_ty

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos
