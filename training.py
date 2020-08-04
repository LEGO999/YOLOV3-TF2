import tensorflow as tf
from ops import decode, yolo_loss, build_boxes, non_max_suppression, draw_boxes_cv2, iou
from data import make_dataset
import os
import datetime
import logging
from model_interface import load_model
import time
import cv2 as cv
from data_augmentation import data_augmentation

# Define gradient
def grad(model,y_true, y_pred, decode_pred, anchors, image_size):
    with tf.GradientTape() as tape:
        loss_value = yolo_loss(y_pred, y_true, decode_pred, anchors, image_size)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(config, anchors):

    num_epoch = int(config.epoch)
    log_dir = './results/'

    # Load dataset
    path = os.getcwd()
    train_file = path + '/train_tf_record'
    test_file = path + '/test_tf_record'

    train_dataset, val_dataset = make_dataset(BATCH_SIZE=config.training_batch_size, file_name=train_file, split=True)
    test_dataset = make_dataset(BATCH_SIZE=config.test_batch_size, file_name=test_file, split=False)

    # Model
    model = load_model(1, anchors, config.model_size, load_full_weights=config.load_full_weights)

    # set optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

    # set Metrics
    tr_sum_loss = tf.keras.metrics.Mean()
    tr_iou = tf.keras.metrics.Mean()

    val_xy_loss = tf.keras.metrics.Mean()
    val_wh_loss = tf.keras.metrics.Mean()
    val_obj_loss = tf.keras.metrics.Mean()
    val_prob_loss = tf.keras.metrics.Mean()
    val_sum_loss = tf.keras.metrics.Mean()
    val_iou = tf.keras.metrics.Mean()

    test_iou = tf.keras.metrics.Mean()

    # Save Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=5)

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = log_dir + current_time
    summary_writer = tf.summary.create_file_writer(tb_log_dir)

    # Restore Checkpoint
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info('Restored from {}'.format(manager.latest_checkpoint))
    else:
        logging.info('Initializing from scratch.')

    # calculate losses, update network and metrics.
    @tf.function
    def train_step(images, y_true):
        # Optimize the model

        with tf.GradientTape() as tape:
            detect0, detect1, detect2 = model(images, training=True, finetuning=config.finetuning)
            de_de0 = decode(detect0, anchors[2], 1, config.model_size)
            de_de1 = decode(detect1, anchors[1], 1, config.model_size)
            de_de2 = decode(detect2, anchors[0], 1, config.model_size)

            loss_de0 = yolo_loss(detect0, y_true, de_de0, anchors[2], config.model_size)
            loss_de1 = yolo_loss(detect1, y_true, de_de1, anchors[1], config.model_size)
            loss_de2 = yolo_loss(detect2, y_true, de_de2, anchors[0], config.model_size)
            total_loss = loss_de0 + loss_de1 + loss_de2
            sum_loss = tf.reduce_sum(total_loss)

        grads = tape.gradient(sum_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        x = tf.concat([de_de0, de_de1, de_de2], axis=1)
        x = build_boxes(x)
        boxes_dicts = non_max_suppression(x, model.n_classes, config.max_out_size, config.iou_threshold,
                                          config.confid_threshold)

        return sum_loss, boxes_dicts

    @tf.function
    def val_step(images, y_true):
        detect0, detect1, detect2 = model(images, training=False)
        de_de0 = decode(detect0, anchors[2], 1, config.model_size)
        de_de1 = decode(detect1, anchors[1], 1, config.model_size)
        de_de2 = decode(detect2, anchors[0], 1, config.model_size)

        loss_de0 = yolo_loss(detect0, y_true, de_de0, anchors[2], config.model_size)
        loss_de1 = yolo_loss(detect1, y_true, de_de1, anchors[1], config.model_size)
        loss_de2 = yolo_loss(detect2, y_true, de_de2, anchors[0], config.model_size)

        total_loss = loss_de0 + loss_de1 + loss_de2

        x = tf.concat([de_de0, de_de1, de_de2], axis=1)
        x = build_boxes(x)
        boxes_dicts = non_max_suppression(x, model.n_classes, config.max_out_size, config.iou_threshold,
                                          config.confid_threshold)

        return total_loss, boxes_dicts

    @tf.function
    def test_step(images):
        detect0, detect1, detect2 = model(images, training=False)
        de_de0 = decode(detect0, anchors[2], 1, config.model_size)
        de_de1 = decode(detect0, anchors[1], 1, config.model_size)
        de_de2 = decode(detect0, anchors[0], 1, config.model_size)

        x = tf.concat([de_de0, de_de1, de_de2], axis=1)
        x = build_boxes(x)
        boxes_dicts = non_max_suppression(x, model.n_classes, config.max_out_size, config.iou_threshold, config.confid_threshold)
        return boxes_dicts

    for epoch in range(num_epoch):
        begin = time.time()

        # Training loop
        for i in train_dataset:
            images = data_augmentation(i[0], config.probability, config.brightness_delta, config.contrast_range, config.hue_delta)
            y_true = i[1:]
            sum_loss, boxes_dicts = train_step(images, y_true)
            tr_sum_loss.update_state(sum_loss)
            pred_points = list(map(lambda x: x[0][...,0:4], boxes_dicts))
            pred_points = tf.concat(pred_points, axis=0)
            for _ in range(config.training_batch_size - tf.shape(pred_points)[0]):
               pred_points = tf.concat(values=[pred_points, tf.constant([[0., 0., 0., 0.]])], axis=0)
            training_iou_batch = tf.reduce_mean(iou(pred_points, tf.cast(tf.stack(y_true[0:4], axis=-1), dtype=tf.float32)))
            tr_iou.update_state(training_iou_batch)

        for j in val_dataset:
            images = j[0]
            y_true = j[1:]
            total_loss, boxes_dicts = val_step(images, y_true)
            val_xy_loss.update_state(total_loss[0])
            val_wh_loss.update_state(total_loss[1])
            val_obj_loss.update_state(total_loss[2])
            val_prob_loss.update_state(total_loss[3])
            val_sum_loss.update_state(tf.reduce_sum(total_loss))

            pred_points = list(map(lambda x: x[0][..., 0:4], boxes_dicts))
            pred_points = tf.concat(pred_points, axis=0)
            for _ in range(config.training_batch_size - tf.shape(pred_points)[0]):
                pred_points = tf.concat(values=[pred_points, tf.constant([[0., 0., 0., 0.]])], axis=0)
            val_iou_batch = tf.reduce_mean(
                iou(pred_points, tf.cast(tf.stack(y_true[0:4], axis=-1), dtype=tf.float32)))
            val_iou.update_state(val_iou_batch)

        with summary_writer.as_default():
            tf.summary.scalar('Train Sum loss', tr_sum_loss.result(), step=epoch)
            tf.summary.scalar('Train IOU', tr_iou.result(), step=epoch)
            tf.summary.scalar('Validation XY loss', val_xy_loss.result(), step=epoch)
            tf.summary.scalar('Validation WH loss', val_wh_loss.result(), step=epoch)
            tf.summary.scalar('Validation OBJ loss', val_obj_loss.result(), step=epoch)
            tf.summary.scalar('Validation PROB loss', val_prob_loss.result(), step=epoch)
            tf.summary.scalar('Validation Sum loss', val_sum_loss.result(), step=epoch)
            tf.summary.scalar('Validation IOU', val_iou.result(), step=epoch)
        end = time.time()
        logging.info("Epoch {:d} Training Sum loss: {:.3}, Training IOU: {:.3%} \n Validation Sum loss: {:.3}, "
                     "Validation IOU: {:.3%}, Time:{:.5}s". format(epoch + 1,tr_sum_loss.result(), tr_iou.result(),
                                                                   val_sum_loss.result(), val_iou.result(), (end - begin)))

        tr_sum_loss.reset_states()
        tr_iou.reset_states()

        val_xy_loss.reset_states()
        val_wh_loss.reset_states()
        val_obj_loss.reset_states()
        val_prob_loss.reset_states()
        val_sum_loss.reset_states()
        val_iou.reset_states()

        if int(ckpt.step) % 5 == 0:
            save_path = manager.save()
            logging.info('Saved checkpoint for epoch {}: {}'.format(int(ckpt.step), save_path))
        ckpt.step.assign_add(1)

        if epoch % 1 == 0:
            for j in test_dataset:
                images = j[0]
                y_true = j[1:]
                boxes_dicts = test_step(images)
                pred_points = list(map(lambda x: x[0][..., 0:4], boxes_dicts))
                pred_points = tf.concat(pred_points, axis=0)
                for _ in range(config.test_batch_size - tf.shape(pred_points)[0]):
                    pred_points = tf.concat(values=[pred_points, tf.constant([[0., 0., 0., 0.]])], axis=0)
                test_iou_batch = tf.reduce_mean(
                    iou(pred_points, tf.cast(tf.stack(y_true[0:4], axis=-1), dtype=tf.float32)))
                test_iou.update_state(test_iou_batch)

            # test image visualization
            time_begin = time.time()
            test_sample = next(iter(test_dataset))
            images = test_sample[0][0]
            y_true = list(map(lambda x: x[0], test_sample[1:]))

            # numpy array is necessary for openCV
            images_tf = tf.expand_dims(images, axis=0)
            images = tf.cast(images * 255, tf.uint8).numpy()
            images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
            box_dicts = test_step(images_tf)
            draw_boxes_cv2(images, box_dicts, [y_true[4].numpy().decode('utf-8')], config.model_size, time_begin)
            cv.rectangle(images, (y_true[0], y_true[1]), (y_true[2], y_true[3]), (0, 0, 255), 2)
            images = cv.cvtColor(images, cv.COLOR_RGB2BGR)
            images = tf.expand_dims(images, axis=0)
            with summary_writer.as_default():
                tf.summary.image('Test Object detection', images, step=epoch)
                tf.summary.scalar('Test IOU', test_iou.result(), step=epoch)
            logging.info("Test IOU: {:.3%}".format(test_iou.result()))
            test_iou.reset_states()
