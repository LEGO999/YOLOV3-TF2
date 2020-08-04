from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from ops import chunk_anchors
from training import train


flags.DEFINE_float('lr',1e-3,'Learning rate of the network.')
flags.DEFINE_integer('epoch', 50 , 'Epoch of training')
flags.DEFINE_integer('training_batch_size', 8 , 'training batch size')
flags.DEFINE_integer('test_batch_size', 16 , 'test batch size')
flags.DEFINE_boolean('finetuning', True, 'If True, only the backend of DNN will be trained. Otherwise, feature extractor will also be trained')
flags.DEFINE_boolean('load_full_weights', False, 'Load full COCO pretrained weights including those of the last detection layers')

flags.DEFINE_multi_integer('model_size', (608, 608), 'Resolution of DNN input, must be the multiples of 32')
flags.DEFINE_integer('max_out_size', 1 , 'maximum detected object amount of one class')
flags.DEFINE_float('iou_threshold', 0.5 , 'threshold of non-max suppression')
flags.DEFINE_float('confid_threshold', 0.5 , 'threshold of confidence')

flags.DEFINE_float('brightness_delta', 0.3, 'brightness_delta of data augmentation')
flags.DEFINE_multi_float('contrast_range', (0.5, 1.5) , 'contrast_range of data augmentation')
flags.DEFINE_float('hue_delta', 0.2 , 'hue_delta of data augmentation, only between (0, 0.5)')
flags.DEFINE_float('probability', 0.8, 'percentage of augmented images')

# Anchors of k-means threshold=0.98
_ANCHORS = [(77.0, 91.0), (89.0, 93.0), (83.0, 101.0), (95.0, 102.0), (92.0, 111.0), (104.0, 108.0), (98.0, 117.0), (110.0, 122.0), (127.0, 134.0)]

# Anchors of k-means threshold=0.99
# _ANCHORS = [(77.0, 92.0), (89.0, 93.0), (83.0, 101.0), (95.0, 101.0), (93.0, 111.0), (105.0, 109.0), (101.0, 119.0), (114.0, 125.0), (151.0, 152.0)]

# Default anchors of COCO
# _ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]

# Anchors of k-means clusters = 3
# _ANCHORS = [(86.0, 98.0), (97.0, 108.0), (110.0, 120.0)]


def main(argv):
    del argv
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    # Anchors[2] bigger than anchors[1], anchors[1] bigger than anchors[0]
    anchors = chunk_anchors(_ANCHORS, 3)

    train(FLAGS, anchors)


if __name__ == '__main__':
    app.run(main)