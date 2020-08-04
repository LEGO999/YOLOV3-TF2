import tensorflow as tf
import os
from model import YOLOV3
from load_weight import load_weights
from absl import logging
from ops import build_boxes, non_max_suppression, decode


def load_model(num_classes, anchors, MODEL_SIZE, load_full_weights):

    model = YOLOV3(n_classes=num_classes, anchors=anchors)
    model.build(input_shape=(None, MODEL_SIZE[0], MODEL_SIZE[1], 3))
    dir = os.getcwd() + '/yolov3.weights'
    load_weights(model.variables, file=dir, load_full_weights=load_full_weights)
    logging.info('Weights are loaded.')

    return model

@tf.function
def inference(model, inputs, anchors, model_size, max_output_size, iou_threshold, confidence_threshold):
    detect0, detect1, detect2 = model(inputs, training=False)
    de_detect0, de_detect1, de_detect2 = decode(detect0, anchors[2], model.n_classes, model_size), \
                                         decode(detect1, anchors[1], model.n_classes, model_size), \
                                         decode(detect2, anchors[0], model.n_classes, model_size)
    x = tf.concat([de_detect0, de_detect1, de_detect2], axis=1)
    x = build_boxes(x)
    boxes_dicts = non_max_suppression(x, model.n_classes, max_output_size, iou_threshold, confidence_threshold)
    return boxes_dicts



