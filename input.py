from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
print(tf.__version__)
import functools

# TODO: Infer number of classes.
# TODO: Make it transferrable across datasets via classes - road, pedastrian, car etc
# def _transform_label(image, label):
#   """two classes: road and background"""
#   print(label.shape)
#   gt_background = tf.reduce_all(label == BACKGROUND_COLOR, axis=2)
#   gt_recip_background = tf.reduce_all(label != BACKGROUND_COLOR, axis=2)
#   gt_background = tf.expand_dims(gt_background, axis=2)
#   gt_recip_background = tf.expand_dims(gt_recip_background, axis=2)
#   label = tf.concat((gt_background, gt_recip_background), axis=2)
#   return image, label

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
  x = tf.argmax(label, axis=2)
  y = tf.abs(x - 1)
  x = tf.expand_dims(x, axis=2)
  y = tf.expand_dims(y, axis=2)
  label = tf.concat([x,y],axis=2)
  return image, label


def get_train_inputs(batch_size, num_classes=None, image_shape=None, repeat=True):
  """
  Return the input function to get the training data
  :param batch_size:  
  :param data: 
  :return: iterator, filename placeholder
  """
  with tf.name_scope('training_data'):
    # Define placeholders
    filename = tf.placeholder(tf.string, shape=[None], name='input')
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
    #dataset.batch(5)
    if repeat:
      dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(ops.GraphKeys.RESOURCES, iterator)
    # sess.run(iterator.initializer, feed_dict={filename: training_filename})
    return iterator, filename


