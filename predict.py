import tensorflow.compat.v1 as tf
import os
import argparse
import datetime
import sys
import numpy as np
import requests
from io import BytesIO
from PIL import Image
tf.disable_eager_execution()

MODEL_SAVE_PATH = "model"
FLAGS = None

def main(_):
    #載入graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+"/crack_captcha.model-19400.meta")
    graph = tf.get_default_graph()
    input_holder = graph.get_tensor_by_name("input/data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("dropout/keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("reshape/predict_max_idx:0")

    content = FLAGS.image
    if FLAGS.image.startswith('http'):
        response = requests.get(FLAGS.image)
        content = Image.open(BytesIO(response.content))
        content.save('images/captcha-test.png')
    else:
        content = Image.open(FLAGS.image)
    im = content.convert('L').resize((120, 100), Image.ANTIALIAS)
    data = np.asarray(im)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        predict = sess.run(predict_max_idx, feed_dict={input_holder: [data], keep_prob_holder : 1.0})
        predictValue = np.squeeze(predict)
        print(predictValue);

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='images/captcha-test.png',
                        help='Image to process')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
