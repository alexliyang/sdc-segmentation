from __future__ import print_function

import tensorflow as tf
import logging
import functools


def _parse_tf_record(example_proto, num_classes=None):
  features = {"height": tf.FixedLenFeature((), tf.int64),
              "width": tf.FixedLenFeature((), tf.int64),
              "image_raw": tf.FixedLenFeature((), tf.string),
              "mask_raw": tf.FixedLenFeature((), tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
  label = tf.decode_raw(parsed_features['mask_raw'], tf.uint8)

  height = tf.cast(parsed_features['height'], tf.int32)
  width = tf.cast(parsed_features['width'], tf.int32)

  image_shape = tf.stack([height, width, 3])
  label_image_shape = tf.stack([height, width, num_classes])
  # restore dim of the image
  image = tf.reshape(image, image_shape)
  label = tf.reshape(label, label_image_shape)

  return image, label

def _resize_image(image, label, image_shape):
  # resize
  image = tf.image.resize_images(image, image_shape)
  label = tf.image.resize_images(label, image_shape)
  return image, label


def get_train_inputs(batch_size, num_classes=None, image_shape=None, repeat=True):
  """
  Return the input function to get the training data
  :param batch_size:  
  :param data: 
  :return: a function that returns (features, labels) when called.
  """
  def train_inputs_from_tf_record():
    with tf.name_scope('training_data'):
      # Define placeholders
      filename = tf.placeholder(tf.string, shape=[None])
      dataset = tf.contrib.data.TFRecordDataset(filename)
      parse_tf_record = functools.partial(
        _parse_tf_record,
        num_classes=num_classes)
      dataset = dataset.map(parse_tf_record)
      # image reshape
      if image_shape:
        resize_fn = functools.partial(
          _resize_image,
          image_shape=image_shape)
        dataset = dataset.map(resize_fn)
      if repeat:
        dataset = dataset.repeat()
      dataset.batch(batch_size)
      iterator = dataset.make_initializable_iterator()
      # sess.run(iterator.initializer, feed_dict={filename: training_filename})
      return iterator, filename


  return train_inputs_from_tf_record

