import tensorflow as tf
from seaborn import color_palette
import numpy as np
import cv2 as cv
import time


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0,0],[pad_beg, pad_end],[pad_beg, pad_end],[0,0]])

    return padded_inputs


class PaddedConv(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=1):
        super().__init__()
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding=('same'if strides == 1 else'valid'), use_bias=False)

    def __call__(self, inputs, training=False):
        if self.strides > 1 :
            x = fixed_padding(inputs, kernel_size=self.kernel_size)
        else:
            x = inputs

        x = self.conv(x)

        return x


class DBL(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides=1):
        super().__init__()
        self.basic0 = PaddedConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.basic1 = tf.keras.layers.BatchNormalization()
        self.basic2 = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, training=False):
        x = self.basic0(inputs)
        x = self.basic1(x, training=training)
        x = self.basic2(x)

        return x


class ResidualBlock(tf.keras.Model):

    def __init__(self, filters, strides=1):
        super().__init__()
        self.basic0 = DBL(filters=filters, kernel_size=1, strides=strides)
        self.basic1 = DBL(filters=2 * filters, kernel_size=3, strides=strides)

    def call(self, inputs, training=False):

        x = self.basic0(inputs, training=training)

        x = self.basic1(x, training=training)

        x = x + inputs

        return x


def upsampling(inputs, output_shape):

    output = tf.image.resize(inputs, (output_shape[2], output_shape[1]), method='nearest')
    return output


def build_boxes(inputs):

    center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = center_x - (width / 2)
    top_left_y = center_y - (height / 2)
    bottom_right_x = center_x + (width / 2)
    bottom_right_y = center_y + (height / 2)

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)
    return boxes


def decode(x, anchors, n_classes, img_size):
    n_anchors = len(anchors)
    shape = x.get_shape().as_list()
    grid_shape = shape[1:3]

    x = tf.reshape(x, [-1, n_anchors * grid_shape[0] * grid_shape[1], n_classes + 5])
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    box_centers, box_shapes, confidence, classes = tf.split(x, [2, 2, 1, n_classes], axis=-1)

    a = tf.range(grid_shape[0], dtype=tf.float32)
    b = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(a, b)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)

    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])

    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])

    box_centers = tf.nn.sigmoid(box_centers)

    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])

    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    output = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    return output

def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):
    batch = tf.unstack(inputs, axis=0)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)

        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)

        boxes = tf.concat([boxes[:, :5], classes], axis=-1)
        boxes_dict = dict()

        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.rank != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores, max_output_size, iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:,:5]
        boxes_dicts.append(boxes_dict)

    return boxes_dicts


def draw_boxes_cv2(img, boxes_dicts, class_names, model_size, time_begin):

    boxes_dicts = boxes_dicts[0]
    colors = (np.array(color_palette("hls", 80)) * 255).astype(np.uint8)
    fontface = cv.FONT_HERSHEY_COMPLEX
    resize_factor = (img.shape[1] / model_size[0], img.shape[0] / model_size[1])
    for cls in range(len(class_names)):
        boxes = boxes_dicts[cls]
        if np.size(boxes) != 0:
            color = tuple(int(i) for i in colors[cls])
            for box in boxes:
                xy, confidence = box[:4], box[4]
                xy = replace_non_finite(xy)
                xy = [xy[i].numpy() * resize_factor[i % 2] for i in range(4)]
                x0, y0, x1, y1= int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                thickness = int((img.shape[0] + img.shape[1]) // 800 )
                cv.rectangle(img, (x0-thickness, y0-thickness), (x1+thickness, y1+thickness), color, thickness)
                text_prob = '{} {:.1f}%'.format(class_names[cls], confidence.numpy() * 100)
                textsize= cv.getTextSize(text_prob, fontFace=fontface, fontScale=0.5, thickness=1)
                cv.rectangle(img, (x0 - thickness, y0), (x0 + textsize[0][0], y0 - textsize[0][1] - 5),color=color, thickness=-1)
                cv.putText(img, text_prob,org=(x0 - 2 * thickness, y0 - 5), fontFace=fontface, fontScale=0.5, color=(255,255,255))
    fps = 1 / (time.time() - time_begin)
    text_time = '{:.1f} fps'.format(fps)
    cv.putText(img, text_time, org=(10, 20), fontFace=fontface, fontScale=0.5,color=(255, 255, 255))
    return fps


def iou(box_1, box_2):
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    iou = tf.expand_dims(int_area / (box_1_area + box_2_area - int_area), axis=-1)
    return iou

def yolo_loss(y_pred, y_true, decode_pred, anchors, image_size):

    num_anchors = len(anchors)
    batch_size = y_true[0].get_shape()[0]
    grid_size = tf.cast(tf.shape(y_pred)[1:3], tf.float32)
    strides = image_size / grid_size

    y_pred = tf.reshape(y_pred, [batch_size, grid_size[0], grid_size[1], num_anchors, -1])
    pred_xy, pred_wh, pred_conf, pred_prob = tf.split(y_pred, [2, 2, 1, -1], axis=-1)

    def pred_box(decode_pred, y_pred):
        box = tf.reshape(decode_pred[..., :4], tf.shape(y_pred[...,:4]))
        x_min = tf.expand_dims(box[..., 0] - box[...,2], axis=-1)
        y_min = tf.expand_dims(box[..., 1] - box[..., 3], axis=-1)
        x_max = tf.expand_dims(box[..., 0] + box[..., 2], axis=-1)
        y_max = tf.expand_dims(box[..., 1] + box[..., 3], axis=-1)
        return tf.concat([x_min, y_min, x_max, y_max], axis=-1)
    pred_box = pred_box(decode_pred, y_pred)
    pred_xy = tf.nn.sigmoid(pred_xy)

    x_min = tf.cast(y_true[0], tf.float32)
    y_min = tf.cast(y_true[1], tf.float32)
    x_max = tf.cast(y_true[2], tf.float32)
    y_max = tf.cast(y_true[3], tf.float32)
    label_center_x = (x_max + x_min) / (2 * strides[0])
    label_center_y = (y_max + y_min) / (2 * strides[1])
    label_center = tf.stack([label_center_y, label_center_x], axis=-1)
    label_center_indices = tf.cast(tf.floor(label_center), tf.int64)

    label_center = tf.tile(tf.expand_dims(label_center, axis=1),[1,num_anchors,1])
    label_center = label_center - tf.floor(label_center)

    batch_indices = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int64), axis=-1)
    anchors_indices = tf.constant([[0, 0]]*batch_size, dtype=tf.int64)
    label_center_indices = tf.concat(values=[batch_indices, label_center_indices, anchors_indices], axis=-1)

    label_width = x_max - x_min
    label_height = y_max - y_min
    # smaller box will be larger weighted
    label_area = label_height * label_width
    area_loss_correction = 2 - (label_area / tf.reduce_max(label_area))

    obj_mask = tf.SparseTensor(indices=label_center_indices, values=[1]*batch_size, dense_shape=[batch_size, grid_size[0], grid_size[1],1,1])
    obj_mask_2ch = tf.tile(tf.sparse.to_dense(sp_input=obj_mask), multiples=[1, 1, 1, num_anchors, 2])
    obj_mask_1ch = tf.tile(tf.sparse.to_dense(sp_input=obj_mask), multiples=[1, 1, 1, num_anchors, 1])

    # localization loss
    pred_xy = tf.boolean_mask(pred_xy, obj_mask_2ch)
    pred_xy = tf.reshape(pred_xy, [batch_size, num_anchors, 2])
    xy_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(pred_xy - label_center), axis=[1, 2]) * area_loss_correction)

    # width and height loss
    label_wh = tf.expand_dims(tf.stack(values=[label_width, label_height], axis=-1),axis=1)
    label_wh = tf.tile(label_wh, [1, num_anchors, 1])
    anchors = tf.cast(tf.tile(tf.expand_dims(anchors, axis=0), multiples=[batch_size, 1, 1]), tf.float32)
    label_wh = tf.math.log(label_wh / anchors)

    pred_wh = tf.boolean_mask(pred_wh, obj_mask_2ch)
    pred_wh = tf.reshape(pred_wh, [batch_size, num_anchors, 2])

    wh_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(pred_wh - label_wh), axis=[1, 2]) * area_loss_correction)

    # objectness loss
    true_box = tf.cast(tf.stack(y_true[0:4], axis=-1), tf.float32)
    ignore_thresh = 0.5
    best_iou = tf.map_fn(
        lambda x: tf.reduce_max(iou(x[0], x[1]), axis=-1, keepdims=True),(pred_box, true_box),tf.float32)
    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
    ignore_mask = tf.broadcast_to(ignore_mask, tf.shape(pred_conf))

    pred_conf = tf.nn.sigmoid(pred_conf)
    obj_conf = tf.boolean_mask(pred_conf, obj_mask_1ch)
    obj_score = tf.reduce_sum(tf.keras.losses.binary_crossentropy(obj_conf, 1.))

    noobj_mask = tf.ones_like(obj_mask_1ch) - obj_mask_1ch
    noobj_conf = tf.boolean_mask(pred_conf*ignore_mask, noobj_mask)
    noobj_score = tf.reduce_sum(tf.keras.losses.binary_crossentropy(noobj_conf, 0.))

    obj_loss = tf.reduce_mean(obj_score + noobj_score)

    pred_prob = tf.boolean_mask(pred_prob, mask=obj_mask_1ch)
    prob_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy([1] * batch_size * num_anchors, pred_prob,
                                                    from_logits=True))

    total_loss = tf.stack(values=[xy_loss, wh_loss, obj_loss, prob_loss], axis=0)
    return total_loss


def chunk_anchors(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def replace_non_finite(tensor):
    return tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))
