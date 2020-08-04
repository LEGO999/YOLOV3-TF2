import cv2 as cv
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import time
from model_interface import inference, load_model
from ops import draw_boxes_cv2
from data import load_class_names


flags.DEFINE_boolean('video', False,
                     'If True, video from the specific path will be used. Otherwise, specific camera will be used')
flags.DEFINE_string('video_path', './data/example.mp4',
                    'path of video file)')
flags.DEFINE_integer('cam_num', 0 , 'number of the camera')


flags.DEFINE_multi_integer('output_res', (1280, 720), 'output resolution of video')
flags.DEFINE_string('output_folder','./output/' , 'path folder output video')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')


flags.DEFINE_multi_integer('model_size', (416, 416), 'Resolution of DNN input, must be the multiples of 32')
flags.DEFINE_integer('max_out_size', 10 , 'maximum detected object amount of one class')
flags.DEFINE_float('iou_threshold', 0.5 , 'threshold of non-max suppression')
flags.DEFINE_float('confid_threshold', 0.5 , 'threshold of confidence')
flags.DEFINE_string('classes','classes.txt', 'path of class label text file')


_ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
logging.set_verbosity(logging.INFO)

def main(argv):
    model = load_model(1, _ANCHORS, FLAGS.model_size, FLAGS.max_out_size, FLAGS.iou_threshold,
                       FLAGS.confid_threshold, finetuning=True)
