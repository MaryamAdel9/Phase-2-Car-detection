# example usage: python yolo_video.py -i video.mp4 -o video_out.avi
import argparse
import glob
import time
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s-%(name)s-%(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
                    help="path to input video file")
parser.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
parser.add_argument("-d", "--display", type=int, default=1,
                    help="display output or not (1/0)")
parser.add_argument("-ht", "--height", type=int, default=1200,
                    help="height of output")
parser.add_argument("-wt", "--width", type=int, default=700,
                    help="width of output")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="confidence threshold")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
                    help="non-maximum supression threshold")

args = parser.parse_args()
logger.info("Parsed Arguments")

CONFIDENCE_THRESHOLD = args.confidence
NMS_THRESHOLD = args.threshold
if not Path(args.input).exists():
    raise FileNotFoundError("Path to video file is not exist.")

vc = cv2.VideoCapture(args.input)
weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]

logger.info("Using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

class_names = list()
with open(labels, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

