import tensorflow as tf
import pandas as pd
import sys
from absl import logging

logging.set_verbosity(logging.INFO)


def generate_tf_records(img_dir, csv_dir, file_name, resolution):
    logging.info('Generating {}'.format(file_name))
    writer = tf.io.TFRecordWriter(file_name)

    def preprocessing(img_path, resolution):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img, dtype=tf.uint8)
        img = tf.image.crop_to_bounding_box(img, 0, 266, 2848, 3426)
        img = tf.image.pad_to_bounding_box(img, 289, 0, 3426, 3426)
        img = tf.image.resize(img, resolution, antialias=True)
        img = tf.cast(img, tf.uint8)
        img = tf.io.encode_jpeg(img, quality=94)
        return img

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    # separate different classes
    df = pd.read_csv(csv_dir)
    r = 0
    for img_name in df['filename']:
        df_img = df[df['filename'] == img_name]
        image_path = img_dir + '/' + img_name
        img = preprocessing(image_path, resolution)
        resize_x_factor = 3426 / resolution[0]
        resize_y_factor = 3426 / resolution[1]
        r += 1
        if r % 20 == 0 or r == len(df['filename']):
            logging.info('Preprocessing data: {}/{}'.format(r, len(df['filename'])))
            sys.stdout.flush()
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []
        class_list = []
        for cls in df_img['class']:
            for i in df_img.loc[df_img['class'] == cls, 'xmin'].values:
                x_min_list.append(int((i - 266) / resize_x_factor))
            for i in df_img.loc[df_img['class'] == cls, 'ymin'].values:
                y_min_list.append(int((i + 289) / resize_y_factor))
            for i in df_img.loc[df_img['class'] == cls, 'xmax'].values:
                x_max_list.append(int((i-266) / resize_x_factor))
            for i in df_img.loc[df_img['class'] == cls, 'ymax'].values:
                y_max_list.append(int((i + 289) / resize_y_factor))
                # class names need to be added only once.
                class_list.append(cls.encode('utf-8'))
        feature = {'xmin': _int64_feature(x_min_list),
                   'ymin': _int64_feature(y_min_list),
                   'xmax': _int64_feature(x_max_list),
                   'ymax': _int64_feature(y_max_list),
                   'class': _bytes_feature(class_list),
                   'image': _bytes_feature([img.numpy()]),
                   'name': _bytes_feature([img_name.encode('utf-8')])}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
    generate_tf_records('./DiseaseGrading/OriginalImages/TestingSet',
                        './DiseaseGrading/OriginalImages/TestingSet/label.csv',
                        'test_tf_record', (608,608))
    generate_tf_records('./DiseaseGrading/OriginalImages/TrainingSet',
                        './DiseaseGrading/OriginalImages/TrainingSet/label.csv',
                        'train_tf_record', (608,608))




