import tensorflow as tf
from ops import DBL,ResidualBlock, upsampling, decode
from ops import yolo_loss
from data import make_dataset

class DarkNet53(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.basic0 = DBL(filters=32, kernel_size=3)
        self.basic1 = DBL(filters=64, kernel_size=3, strides=2)

        self.multi0 = ResidualBlock(32)
        self.basic2 = DBL(filters=128, kernel_size=3, strides=2)
        self.multi1 = tf.keras.Sequential()
        for _ in range(2):
            self.multi1.add(ResidualBlock(64))
        self.basic3 = DBL(filters=256, kernel_size=3, strides=2)
        self.multi2 = tf.keras.Sequential()
        for _ in range(8):
            self.multi2.add(ResidualBlock(128))
        self.basic4 = DBL(filters=512, kernel_size=3, strides=2)
        self.multi3 = tf.keras.Sequential()
        for _ in range(8):
            self.multi3.add(ResidualBlock(256))
        self.basic5 = DBL(filters=1024, kernel_size=3, strides=2)
        self.multi4 = tf.keras.Sequential()
        for _ in range(4):
            self.multi4.add(ResidualBlock(512))

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training=training)
        x = self.basic1(x, training=training)
        x = self.multi0(x, training=training)
        x = self.basic2(x, training=training)
        x = self.multi1(x, training=training)
        x = self.basic3(x, training=training)
        route0 = self.multi2(x, training=training)
        x = self.basic4(route0, training=training)
        route1 = self.multi3(x, training=training)
        x = self.basic5(route1, training=training)
        output = self.multi4(x, training)

        return route0, route1, output


class YoloConvBlock(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.basic0 = DBL(filters=filters, kernel_size=1)
        self.basic1 = DBL(filters=2 * filters, kernel_size=3)
        self.basic2 = DBL(filters=filters, kernel_size=1)
        self.basic3 = DBL(filters=2 * filters, kernel_size=3)
        self.basic4 = DBL(filters=filters, kernel_size=1)
        self.basic5 = DBL(filters=2 * filters, kernel_size=3)

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training)
        x = self.basic1(x, training)
        x = self.basic2(x, training)
        x = self.basic3(x, training)
        route = self.basic4(x, training)
        output = self.basic5(route, training)

        return route, output


class DetectionLayer(tf.keras.Model):
    def __init__(self, n_classes, anchors):
        super().__init__()
        self.n_classes = n_classes
        self.n_anchors = len(anchors)
        self.conv = tf.keras.layers.Conv2D(filters=self.n_anchors * (self.n_classes + 5), kernel_size=1, strides=1)

    def call(self, inputs, training=False):

        x = self.conv(inputs)

        return x

class YOLOV3(tf.keras.Model):
    def __init__(self, n_classes, anchors):
        super().__init__()
        self.n_classes = n_classes
        self.Feature_extractor = DarkNet53()
        self.Conv_Block0 = YoloConvBlock(filters=512)
        self.Conv_Block1 = YoloConvBlock(filters=256)
        self.Conv_Block2 = YoloConvBlock(filters=128)
        self.DBL0 = DBL(filters=256, kernel_size=1)
        self.DBL1 = DBL(filters=128, kernel_size=1)
        self.Detector0 = DetectionLayer(n_classes, anchors=anchors[2])
        self.Detector1 = DetectionLayer(n_classes, anchors=anchors[1])
        self.Detector2 = DetectionLayer(n_classes, anchors=anchors[0])

    def call(self, inputs, training=False, finetuning=True):
        if finetuning:
            self.Feature_extractor.trainable = False
        else:
            self.Feature_extractor.trainable = True
        route0, route1, x = self.Feature_extractor(inputs, training=False)
        route, x = self.Conv_Block0(x, training=training)
        detect0 = self.Detector0(x, training=training)

        x = self.DBL0(route, training=training)
        upsample_size = route1.get_shape().as_list()
        x = upsampling(x, output_shape=upsample_size)
        x = tf.concat([x, route1], axis=-1)
        route, x = self.Conv_Block1(x, training=training)
        detect1 = self.Detector1(x, training=training)

        x = self.DBL1(route, training=training)
        upsample_size = route0.get_shape().as_list()
        x = upsampling(x, output_shape=upsample_size)
        x = tf.concat([x, route0], axis=-1)
        _, x = self.Conv_Block2(x, training=training)
        detect2 = self.Detector2(x, training=training)

        return detect0, detect1, detect2



if __name__ == '__main__':

    _ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
    model = YOLOV3(n_classes=1, anchors=_ANCHORS)
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)


    test_set = make_dataset(BATCH_SIZE=8, file_name='test_tf_record', split=False)

    for i in test_set:
        images = i[0]
        image_size = tf.cast(tf.shape(images)[1:3], tf.float32)
        label = i[1:]
        images = tf.cast(images, tf.float32) / 255.0
        detect0, detect1, detect2 = model(images, training=True, finetuning=True)
        de_de0 = decode(detect0, _ANCHORS[6:9], 1, image_size)
        total_loss = yolo_loss(detect0, label, de_de0, _ANCHORS[6:9], image_size)
        break
