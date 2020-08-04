import tensorflow as tf
import matplotlib.pyplot as plt
import random


@tf.function
def data_augmentation(image, probablility, brightness_delta=0.3, contrast_range=(0.5, 1.5), hue_delta=0.1):
    random_prob = random.uniform(0, 1)
    if random_prob >= 1 - probablility:
        img = tf.image.random_brightness(image, max_delta=brightness_delta)
        img = tf.image.random_contrast(img, lower=contrast_range[0], upper=contrast_range[1])
        img = tf.image.random_hue(img, max_delta=hue_delta)
        return img
    else:
        return image


if __name__ == '__main__':
    img = tf.io.read_file('./DiseaseGrading/OriginalImages/TrainingSet/IDRiD_001.jpg')
    img = tf.io.decode_image(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    img = data_augmentation(img, probablility=1, brightness_delta=0.4, hue_delta=0.2)
    img = tf.squeeze(img)
    plt.imshow(img)
    plt.show()