import tensorflow as tf
from absl import logging

def make_folder_dataset(img_dir, batch_size=1):

    list_ds = tf.data.Dataset.list_files(str(img_dir + '/*'))
    name_ds = [i.numpy() for i in list_ds]
    def parse_image(filename):

      image = tf.io.read_file(filename)
      image = tf.image.decode_jpeg(image)
      image = tf.image.convert_image_dtype(image, tf.float32)
      return image

    img_ds = list_ds.map(parse_image).batch(batch_size)

    return img_ds, name_ds


def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def make_dataset(BATCH_SIZE, file_name, split=False, split_train_size=0.9):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    raw_dataset = tf.data.TFRecordDataset([file_name])
    feature_description = {'xmin': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                           'ymin': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                           'xmax': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                           'ymax': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                           'class': tf.io.FixedLenFeature([], tf.string, default_value=''),
                           'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
                           'name': tf.io.FixedLenFeature([], tf.string, default_value='')}

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    if not split:
        count = 0
        for _ in parsed_dataset:
            count += 1
        dataset = parsed_dataset.map(lambda x: (
        tf.io.decode_image(x['image'], dtype=tf.float32), x['xmin'], x['ymin'], x['xmax'], x['ymax'], x['class'],
        x['name']))
        test_set = dataset.prefetch(buffer_size=AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True)
        logging.info('Non-split dataset is made.')
        return test_set

    if split:
        count = 0
        for _ in parsed_dataset:
            count += 1
        dataset = parsed_dataset.map(lambda x: (
        tf.io.decode_image(x['image'], dtype=tf.float32), x['xmin'], x['ymin'], x['xmax'], x['ymax'], x['class'],
        x['name'])).shuffle(buffer_size=count)
        train_size = int(count * split_train_size)
        train_set = dataset.take(train_size)
        valid_set = dataset.skip(train_size)
        train_set = train_set.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        valid_set = valid_set.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        logging.info('Two split datasets are made.')
        return train_set, valid_set

if __name__ == '__main__':
    # a = make_dataset('CAR', [416,416], 32)
    #
    # def show(image):
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()
    #
    # for batch in a:
    #     for i in batch:
    #         print(i)
    # train_set = make_dataset(BATCH_SIZE=1, file_name='train_tf_record', split=False)
    test_set = make_dataset(BATCH_SIZE=1, file_name='test_tf_record', split=False)
    for num, i in enumerate(test_set):
        images = tf.squeeze(i[0]).numpy()
        y_true = i[1:]
        # print(y_true[5])
        points = tf.concat(y_true[0:4], axis=0)
        tf.debugging.assert_greater_equal(
            points, tf.zeros_like(points), message=y_true[5], summarize=None, name=None
        )
        # cv.rectangle(images, (y_true[0], y_true[1]), (y_true[2], y_true[3]), (0, 0, 255), 2)
        # plt.imshow(images)
        # plt.show()

    # a = load_class_names('classes.txt')
    # print(a)