from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import load_darknet_weights

flags.DEFINE_string('weights', 'weights/yolov3_custom_last.weights', 'path to weights file')
flags.DEFINE_string('output', 'weights/yolov3-custom.tf', 'path to output')
flags.DEFINE_integer('num_classes', 3, 'number of classes in the model')


def main(_argv):
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
