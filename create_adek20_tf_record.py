from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from glob import glob
import re
import numpy as np
import scipy.misc
import pandas as pd


import tensorflow as tf

"""
Generate tf_record for ADEK20 Challange
"""

flags = tf.app.flags
flags.DEFINE_string('training_path', '', 'training_path')
flags.DEFINE_string('label_path', '', 'label_path')
flags.DEFINE_string('image_file', '', 'image_file')
flags.DEFINE_string('output_file', '', 'Path to output data')
FLAGS = flags.FLAGS

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_tf_examples(image_paths, label_paths):
  for image_path, label_path in zip(image_paths, label_paths):
    print(image_path)
    image = scipy.misc.imread(image_path, mode='RGB')
    gt_image = scipy.misc.imread(label_path, mode='RGB')
    # one-hot encoding
    gt_image = gt_image[:, :, 0]
    height, width, _ = image.shape
    img_raw, ann_raw = image.flatten().tostring(), gt_image.flatten().tostring()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'image_raw': _bytes_feature(img_raw),
			'mask_raw': _bytes_feature(ann_raw)}))
    yield tf_example

def main(_):
  _image_path = FLAGS.training_path
  _label_path = FLAGS.label_path
  _file_path = FLAGS.image_file
  image_names = [img.split('/')[1] for img in pd.read_csv(_file_path, header=None)[0].tolist()]
  image_paths = [_image_path + image_name for image_name in image_names]
  label_paths = [_label_path + image_name.replace(".jpg",".png") for image_name in image_names]

  writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
  for tf_example in generate_tf_examples(image_paths, label_paths):
    writer.write(tf_example.SerializeToString())
  writer.close()

if __name__ == '__main__':
  tf.app.run()