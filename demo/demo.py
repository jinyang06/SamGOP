# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import detection_utils as utils
from maskGOP import add_maskdino_config
from predictor import VisualizationDemo
from PIL import Image,ImageDraw
from tqdm import tqdm
import json


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/data1/jinyang/TransGOP/MaskDINO2/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    file_dir = '/data1/jinyang/TransGOP/MaskDINO2/datasets/coco/val2017_real/'
    # save_dir = '/data1/jinyang/TransGOP/MaskDINO2/demo/gooreal_vis_heatmap_pretrain/'
    # save_dir_cone = '/data1/jinyang/TransGOP/MaskDINO2/demo/gooreal_vis_gazecone_pretrain/'
    save_dir = '/data1/jinyang/TransGOP/MaskDINO2/demo/gooreal_only_box/'

    save_dir_gos = '/data1/jinyang/TransGOP/MaskDINO2/demo/gooreal_only_box/'
    if not os.path.exists(save_dir_gos):
        os.mkdir(save_dir_gos)
    # if not os.path.exists(save_dir_cone):
    #     os.mkdir(save_dir_cone)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_list = os.listdir(file_dir)

    ann_file_path = '/data1/jinyang/TransGOP/MaskDINO2/datasets/coco/annotations/instances_val2017_real.json'
    f = open(ann_file_path, 'r')
    ann_file_dict = json.load(f)

    for index, filename in enumerate(tqdm(file_list)):
        args.input = file_dir + filename
        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            path = args.input

            path_list = path.split('/')
            img_name = path_list[-1]

            ann = ann_file_dict['images']
            for index, item in enumerate(ann):
                if item['file_name'] == img_name:
                    break
            gaze_ann = item['gaze_related_ann']
            eye_point_gt = [gaze_ann['hx'] / 640 * 1920, gaze_ann['hy'] / 480 * 1080]
            gaze_point_gt = [gaze_ann['gaze_cx'] / 640 * 1920, gaze_ann['gaze_cy'] / 480 * 1080]
            # for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            # predictions, visualized_output, heatmap, heads, visualized_gos_output = demo.run_on_image(img)
            predictions, visualized_output, heatmap, heads = demo.run_on_image(img)
            pred_head_box = heads[0]['boxes'].cpu().numpy()
            pred_head_box[:,[0, 2]] = pred_head_box[:,[0, 2]] /224 *1920
            pred_head_box[:, [1, 3]] = pred_head_box[:, [1, 3]] / 224 * 1080
            pred_head_box = pred_head_box[0]
            pred_head_box[0] = pred_head_box[0] + 200
            pred_head_box[2] = pred_head_box[2] - 200
            pred_head_box[1] = pred_head_box[1] + 100
            pred_head_box[3] = pred_head_box[3] - 100
            cor_heatmap = heatmap[0][0].cpu().numpy()
            h_index, w_index = np.unravel_index(cor_heatmap.argmax(), cor_heatmap.shape)
            pred_gaze_x, pred_gaze_y = [w_index / 64 * 1920, h_index / 64 * 1080]
            eye_point_gt_ = (int(eye_point_gt[0]), eye_point_gt[1])  # 起始点坐标
            pred_gaze_point = (int(pred_gaze_x), int(pred_gaze_y))

            im = visualized_output.get_image()[:, :, ::-1]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            # im.show()
            # x=0
            # im =Image.open(path)
            # hm = heatmap[0][0].cpu().numpy()
            # hm = hm*225
            # hm[hm>255] = 255
            # im = hm.astype(np.uint8)
            # im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
            # im = cv2.resize(im, (1920, 1080))

            # im = visualized_gos_output.get_image()[:, :, ::-1]
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # im = Image.fromarray(im)

            # im = Image.new('RGB', (1920, 1080), (255, 255, 255))

            # im_with_hm = utils.generate_attention_map(im, heatmap[0][0])

            # draw_im = ImageDraw.Draw(im_with_hm)
            # draw_im.rectangle(pred_head_box.tolist(),fill=None,outline='yellow', width=6)
            # draw_im.line((eye_point_gt_[0], eye_point_gt_[1], pred_gaze_point[0], pred_gaze_point[1]), fill='yellow',
            #           width=5)
            # draw_im.ellipse(
            #     (pred_gaze_point[0] - 7, pred_gaze_point[1] - 7, pred_gaze_point[0] + 7, pred_gaze_point[1] + 7),
            #     fill='yellow')
            # draw_im.ellipse((eye_point_gt_[0] - 7, eye_point_gt_[1] - 7, eye_point_gt_[0] + 7, eye_point_gt_[1] + 7),
            #              fill='yellow')
            # im_with_cone = utils.generate_attention_map_(im, gaze_cone[0][0])
            # im_with_attention = utils.generate_attention_map_(im, attention[0][0])
            # draw = ImageDraw.Draw(im_with_hm)
            # draw.ellipse([gaze_point_gt[0] - 8, gaze_point_gt[1] - 8, gaze_point_gt[0] + 8, gaze_point_gt[1] + 8],
            #              fill='green', outline=None)
            # im_with_hm.save(save_dir+filename)
            # im_with_cone.save(save_dir_cone + filename)
            # im_with_attention.save(save_dir_attention + filename)
            im.save(save_dir + filename)
            # cv2.imwrite(save_dir + filename, im)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                x=0
                # cv2.imwrite('./vis/'+filename, visualized_output.get_image()[:, :, ::-1])
                # eye_point_gt_ = (int(eye_point_gt[0]), eye_point_gt[1])  # 起始点坐标
                # pred_gaze_point = (int(pred_gaze_x), int(pred_gaze_y))
                # results_img = visualized_gos_output.get_image()[:, :, ::-1]
                # results_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
                # img_r = Image.fromarray(results_img)
                # draw = ImageDraw.Draw(img_r)
                # draw.rectangle(pred_head_box.tolist(), fill=None, outline='green', width=5)
                # draw.line((eye_point_gt_[0], eye_point_gt_[1], pred_gaze_point[0], pred_gaze_point[1]),fill='red',width=5)
                # draw.line((eye_point_gt_[0], eye_point_gt_[1], gaze_point_gt[0], gaze_point_gt[1]), fill='green',
                #           width=5)
                # draw.ellipse((gaze_point_gt[0] - 7, gaze_point_gt[1] - 7, gaze_point_gt[0] + 7, gaze_point_gt[1] + 7),
                #              fill='green')
                # draw.ellipse((eye_point_gt_[0]-7, eye_point_gt_[1]-7, eye_point_gt_[0]+7, eye_point_gt_[1]+7),fill='red')
                # draw.ellipse((pred_gaze_point[0] - 7, pred_gaze_point[1] - 7, pred_gaze_point[0] + 7, pred_gaze_point[1] + 7),fill='red')
                # img_r.save(save_dir_gos + filename)

                # 在图像上画线
                  # 终点坐标

                # cv2.imwrite(save_dir_gos + filename, results_img)
                    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    # if cv2.waitKey(0) == 27:
                    #     break  # esc to quit
        elif args.webcam:
            assert args.input is None, "Cannot have both --input and --webcam!"
            assert args.output is None, "output not yet supported with --webcam!"
            cam = cv2.VideoCapture(0)
            for vis in tqdm.tqdm(demo.run_on_video(cam)):
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            cam.release()
            cv2.destroyAllWindows()
        elif args.video_input:
            video = cv2.VideoCapture(args.video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(args.video_input)
            codec, file_ext = (
                ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                warnings.warn("x264 codec not available, switching to mp4v")
            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(args.video_input)
            for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                if args.output:
                    output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            video.release()
            if args.output:
                output_file.release()
            else:
                cv2.destroyAllWindows()
