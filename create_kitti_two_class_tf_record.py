from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
from glob import glob
import re
import numpy as np
import scipy.misc


import tensorflow as tf

BACKGROUND_COLOR = np.array([255, 0, 0])


"""
Generate tf_record for Kitti segmentation dataset
"""

flags = tf.app.flags
flags.DEFINE_string('output_file', '', 'TFRecord')
flags.DEFINE_string('input_path', '', 'Path to input data')
FLAGS = flags.FLAGS

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate_tf_examples(image_paths, label_paths):
  for image_path, label_path in zip(image_paths, label_paths):
    image = scipy.misc.imread(image_path, mode='RGB')
    gt_image_file = label_paths[os.path.basename(image_path)]
    gt_image = scipy.misc.imread(gt_image_file, mode='RGB')

    gt_background = np.all(gt_image == BACKGROUND_COLOR, axis=2)
    gt_background = gt_background.reshape(*gt_background.shape, 1)
    gt_image = np.concatenate((gt_background, np.invert(gt_background)), axis=2)

    height, width, _ = image.shape

    img_raw, ann_raw = image.flatten().tostring(), gt_image.flatten().tostring()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'image_raw': _bytes_feature(img_raw),
      'mask_raw': _bytes_feature(ann_raw)}))
    yield tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
  _input_path = FLAGS.input_path
  image_paths = glob(os.path.join(_input_path, 'image_2', '*.png'))
  # take only images with suffixes *road*
  label_paths = {
		re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
		for path in glob(os.path.join(_input_path, 'gt_image_2', '*_road_*.png'))
	}
  for tf_example in generate_tf_examples(image_paths, label_paths):
    writer.write(tf_example.SerializeToString())
  writer.close()

if __name__ == '__main__':
  tf.app.run()