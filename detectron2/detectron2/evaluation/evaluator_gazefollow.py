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

from torch import Tensor
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_area)
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_iou)
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


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
        all_gazepoints = []
        all_predmap = []
        all_gtmap = []
        all_min_dist = []
        all_avg_dist = []

        for idx, inputs in enumerate(data_loader):

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            head_box_gt = inputs[0]['gaze_related_ann']['head_box']
            head_box_gt = torch.tensor(head_box_gt, dtype=torch.float32, device='cuda')
            head_box_pred = outputs[3][0]['boxes']
            ious = box_iou(
                head_box_gt,
                head_box_pred,
            )
            ious = ious.cpu().numpy()

            gaze_heatmap_pred = outputs[1]
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(0).squeeze(0).cpu().numpy()

            num_heatmap_list = []
            for index, arr in enumerate(ious):
                index =np.argmax(arr)
                heatmap = gaze_heatmap_pred[index]
                num_heatmap_list.append(heatmap)
            # gaze_direction_pred = np.array(outputs[2].squeeze(0).cpu())
            outputs = outputs[0]
            #################################################
            # gaze metric
            #################################################
            # gaze_direction_gt = inputs[0]['gaze_direction']
            # cosine_similarity = np.dot(gaze_direction_pred, gaze_direction_gt)
            # angle_error = np.arccos(cosine_similarity) * 180 / np.pi
            # all_direction_error.append(angle_error)
            gaze_heatmap_gt = inputs[0]['gaze_heatmap']
            eye_point_set = inputs[0]['eye']
            gaze_point_set = inputs[0]['gaze']

            for eye, gaze, hm_gt, hm_pred in zip(eye_point_set, gaze_point_set, gaze_heatmap_gt, num_heatmap_list):
                eye_point = np.array(eye)
                single_person_dist = []
                for gaze_ in gaze:
                    f_point = hm_pred
                    gt_point = np.array(gaze_)
                    out_size = 64  # Size of heatmap for chong output
                    f_point = f_point.reshape([out_size, out_size])
                    h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
                    f_point = np.array([w_index / out_size, h_index / out_size])
                    f_error = f_point - gt_point
                    f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)
                    single_person_dist.append(f_dist)

                    gt_heatmap = np.zeros((5, 5))
                    x, y = list(map(int, gt_point * 5))
                    gt_heatmap[y, x] = 1.0

                heatmap = np.copy(hm_pred)
                heatmap = np.squeeze(heatmap)
                heatmap = cv2.resize(heatmap, (5, 5))

                all_predmap.append(heatmap)
                all_gtmap.append(gt_heatmap)
                all_min_dist.append(min(single_person_dist))
                all_avg_dist.append(sum(single_person_dist) / len(single_person_dist))


                #AUC

                # w, h = out_res
                # target_map = np.zeros((h, w))
                # for p in gaze_pts:
                #     if p[0] >= 0:
                #         x, y = map(int, [p[0] * w.float(), p[1] * h.float()])
                #         x = min(x, w - 1)
                #         y = min(y, h - 1)
                #         target_map[y, x] = 1




            # f_point = gaze_heatmap_pred
            # gt_point = np.array(inputs[0]['gaze'])
            # eye_point = np.array(inputs[0]['eye'])
            # out_size = 64  # Size of heatmap for chong output
            # heatmap = np.copy(f_point)
            # f_point = f_point.reshape([out_size, out_size])
            #
            # h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            # f_point = np.array([w_index / out_size, h_index / out_size])
            # f_error = f_point - gt_point
            # f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)
            #
            # # angle
            # f_direction = f_point - eye_point
            # gt_direction = gt_point - eye_point
            #
            # norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
            # norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5
            #
            # f_cos_sim = (f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]) / \
            #             (norm_gt * norm_f + 1e-6)
            # f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            # f_angle = np.arccos(f_cos_sim) * 180 / np.pi
            #
            # # AUC calculation
            # heatmap = np.squeeze(heatmap)
            # heatmap = cv2.resize(heatmap, (5, 5))
            # gt_heatmap = np.zeros((5, 5))
            # x, y = list(map(int, gt_point * 5))
            # gt_heatmap[y, x] = 1.0
            # all_gazepoints.append(f_point)
            # all_predmap.append(heatmap)
            # all_gtmap.append(gt_heatmap)
            # total_error.append([f_dist, f_angle])
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

        min_dist = np.mean(np.array(all_min_dist))
        avg_dist = np.mean(np.array(all_avg_dist))
        # ang_error = np.mean(np.array(all_direction_error))
        # all_gazepoints = np.vstack(all_gazepoints)
        all_predmap = np.stack(all_predmap).reshape([-1])
        all_gtmap = np.stack(all_gtmap).reshape([-1])
        auc = roc_auc_score(all_gtmap, all_predmap)

        logger.info('Gaze estimation evaluation metric:')
        logger.info('AUC: {}'.format(auc))
        logger.info('Min Dist: {}'.format(min_dist))
        logger.info('Avg Dist: {}'.format(avg_dist))
        # logger.info('Direction Angle error: {}'.format(ang_error))

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
