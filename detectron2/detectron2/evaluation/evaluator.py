# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

import numpy as np
import cv2
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from thop import profile

def calculate_iou_(boxes1, boxes2):
    # 获取左上角和右下角坐标
    x1_tl, y1_tl, x1_br, y1_br = np.split(boxes1, 4, axis=1)
    x2_tl, y2_tl, x2_br, y2_br = np.split(boxes2, 4, axis=1)

    # 计算相交区域的坐标
    x_intersection = np.maximum(x1_tl, x2_tl)
    y_intersection = np.maximum(y1_tl, y2_tl)
    x_intersection_right = np.minimum(x1_br, x2_br)
    y_intersection_bottom = np.minimum(y1_br, y2_br)

    # 计算相交区域的宽度和高度
    w_intersection = np.maximum(0, x_intersection_right - x_intersection)
    h_intersection = np.maximum(0, y_intersection_bottom - y_intersection)

    # 计算相交区域和并集的面积
    area_intersection = w_intersection * h_intersection
    area_union = (x1_br - x1_tl) * (y1_br - y1_tl) + (x2_br - x2_tl) * (y2_br - y2_tl) - area_intersection

    # 计算IoU
    iou = area_intersection / np.maximum(area_union, 1e-10)

    return iou
# def calculate_iou_(box1, box2):
#     # 获取矩形框的坐标
#     x1_tl, y1_tl, x1_br, y1_br = box1
#     x2_tl, y2_tl, x2_br, y2_br = box2
#
#     # 计算相交区域的坐标
#     x_intersection = max(x1_tl, x2_tl)
#     y_intersection = max(y1_tl, y2_tl)
#     x_intersection_right = min(x1_br, x2_br)
#     y_intersection_bottom = min(y1_br, y2_br)
#
#     # 计算相交区域的宽度和高度
#     w_intersection = max(0, x_intersection_right - x_intersection)
#     h_intersection = max(0, y_intersection_bottom - y_intersection)
#
#     # 计算相交区域和并集的面积
#     area_intersection = w_intersection * h_intersection
#     area_union = (x1_br - x1_tl) * (y1_br - y1_tl) + (x2_br - x2_tl) * (y2_br - y2_tl) - area_intersection
#
#     # 计算IoU
#     iou = area_intersection / max(area_union, 1e-10)
#
#     return iou


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results




def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        #######################################
        # gaze metric
        #######################################
        l2_threshold = 0.15
        true_positives = 0
        false_positives = 0

        all_gazepoints = []

        all_predmap = []
        all_gtmap = []
        all_direction_error = []
        total_error = []

        all_predmap_gtr = []
        all_gtmap_gtr = []
        total_error_gtr = []

        for idx, inputs in enumerate(data_loader):

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            # start_time_ = time.time()
            # flops, params = profile(model, inputs=[inputs])
            # end_time_ = time.time()
            # inference_time = end_time_ - start_time_

            outputs = model(inputs)
            head_box = outputs[-1][0]['boxes'].cpu().numpy()
            gt_head_box = inputs[0]['gaze_related_ann']['head_box'][:, 0:4]
            head_box_iou = calculate_iou_(head_box, gt_head_box)
            head_box_iou = head_box_iou[0]
            gaze_direction_pred = np.array(outputs[2].squeeze(0).cpu())

            gaze_heatmap_pred = outputs[1]
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(0).squeeze(0).cpu().numpy()
            outputs = outputs[0]
            #################################################
            # gaze metric
            #################################################

            gaze_direction_gt = inputs[0]['gaze_direction']
            cosine_similarity = np.dot(gaze_direction_pred, gaze_direction_gt)
            angle_error = np.arccos(cosine_similarity) * 180 / np.pi
            all_direction_error.append(angle_error)

            gaze_heatmap_gt = inputs[0]['gaze_heatmap']
            f_point = gaze_heatmap_pred
            gt_point = np.array(inputs[0]['gaze'])
            eye_point = np.array(inputs[0]['eye'])
            out_size = 64  # Size of heatmap for chong output
            heatmap = np.copy(f_point)
            f_point = f_point.reshape([out_size, out_size])

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            f_point = np.array([w_index / out_size, h_index / out_size])
            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            # angle
            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
            norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5

            f_cos_sim = (f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]) / \
                        (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi

            # AUC calculation
            heatmap = np.squeeze(heatmap)
            heatmap = cv2.resize(heatmap, (5, 5))
            gt_heatmap = np.zeros((5, 5))
            x, y = list(map(int, gt_point * 5))
            gt_heatmap[y, x] = 1.0

            if f_dist < l2_threshold and head_box_iou > 0.5:
                true_positives += 1
                all_predmap_gtr.append(heatmap)
                all_gtmap_gtr.append(gt_heatmap)
                total_error_gtr.append([f_dist, f_angle])
            else:
                false_positives += 1

            all_gazepoints.append(f_point)
            all_predmap.append(heatmap)
            all_gtmap.append(gt_heatmap)
            total_error.append([f_dist, f_angle])
            #############################################

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

        l2, ang = np.mean(np.array(total_error), axis=0)
        ang_error = np.mean(np.array(all_direction_error))
        # all_gazepoints = np.vstack(all_gazepoints)
        all_predmap = np.stack(all_predmap).reshape([-1])
        all_gtmap = np.stack(all_gtmap).reshape([-1])
        auc = roc_auc_score(all_gtmap, all_predmap)

        logger.info('Gaze estimation evaluation metric:')
        logger.info('AUC: {}'.format(auc))
        logger.info('Dist: {}'.format(l2))
        logger.info('Ang: {}'.format(ang))
        logger.info('Direction Angle error: {}'.format(ang_error))

        l2_gtr, ang_gtr = np.mean(np.array(total_error_gtr), axis=0)
        all_predmap_gtr = np.stack(all_predmap_gtr).reshape([-1])
        all_gtmap_gtr = np.stack(all_gtmap_gtr).reshape([-1])
        auc_gtr = roc_auc_score(all_gtmap_gtr, all_predmap_gtr)
        AP = true_positives/(true_positives+false_positives)
        logger.info('Gaze estimation evaluation metric using GTR:')
        logger.info('AUC_GTR: {}'.format(auc_gtr))
        logger.info('Dist_GTR: {}'.format(l2_gtr))
        logger.info('Ang_GTR: {}'.format(ang_gtr))
        logger.info('mAP: {}'.format(AP))
        logger.info('True_positives: {}'.format(true_positives))
        logger.info('False_positives: {}'.format(false_positives))

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
