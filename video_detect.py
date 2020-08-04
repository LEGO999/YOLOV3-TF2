import cv2 as cv
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import time
from model_interface import inference, load_model
from ops import draw_boxes_cv2, chunk_anchors
from data import load_class_names
import statistics


flags.DEFINE_boolean('video', True,
                     'If True, video from the specific path will be used. Otherwise, specific camera will be used')
flags.DEFINE_string('video_path', './data/example1.mp4',
                    'path of video file)')
flags.DEFINE_integer('cam_num', 0 , 'number of the camera')


flags.DEFINE_multi_integer('output_res', (1920, 1080), 'output resolution of video')
flags.DEFINE_string('output_folder','./output/' , 'path folder output video')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')


flags.DEFINE_multi_integer('model_size', (608, 608), 'Resolution of DNN input, must be the multiples of 32')
flags.DEFINE_integer('max_out_size', 20 , 'maximum detected object amount of one class')
flags.DEFINE_float('iou_threshold', 0.4 , 'threshold of non-max suppression')
flags.DEFINE_float('confid_threshold', 0.3 , 'threshold of confidence')
flags.DEFINE_string('classes','classes.txt', 'path of class label text file')

_ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
logging.set_verbosity(logging.INFO)

def main(argv):
    del argv
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    anchors = chunk_anchors(_ANCHORS, 3)

    if FLAGS.video:
        cap = cv.VideoCapture(FLAGS.video_path)
        output_file = FLAGS.video_path.split('/')
        for i in output_file:
            if '.mp4' in i:
                output_file = i.rstrip('.mp4')
        output_file = FLAGS.output_folder + output_file + '_' + time.strftime('%m%d%H%M') + '.mp4'

    else:
        cap = cv.VideoCapture(FLAGS.cam_num)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FPS, 30)
        output_file = FLAGS.output_folder + 'SELFIE' + '_' + time.strftime('%m%d%H%M') + '.mp4'

    fourcc = cv.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv.VideoWriter(output_file, fourcc, 15.0, (FLAGS.output_res[0], FLAGS.output_res[1]))
    if not cap.isOpened():
        logging.error("Cannot get streaming")
        exit()
    classes = load_class_names(FLAGS.classes)
    n_classes= len(classes)
    model = load_model(n_classes, anchors, FLAGS.model_size, True)
    logging.info('Video frames are being captured.')
    fps_list = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.resize(frame, tuple(FLAGS.output_res))
        time_begin = time.time()
        frame_tf = tf.expand_dims(frame, axis=0)
        frame_tf = tf.cast(frame_tf, dtype=tf.float32)
        frame_tf = frame_tf / 255.0
        frame_tf = tf.image.resize(frame_tf, size=FLAGS.model_size)
        detections = inference(model, frame_tf, anchors, FLAGS.model_size, FLAGS.max_out_size, FLAGS.iou_threshold, FLAGS.confid_threshold)
        fps = draw_boxes_cv2(frame, detections, classes, FLAGS.model_size, time_begin)
        fps_list.append(fps)
        out.write(frame)
        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything is done, release the capture
    cap.release()
    out.release()
    cv.destroyAllWindows()
    average_fps = statistics.mean(fps_list)
    logging.info('Average FPS is {:.1f}'.format(average_fps))


if __name__ == '__main__':
    app.run(main)