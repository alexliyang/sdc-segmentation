
import os
import sys

import tensorflow as tf
import utils


sys.path.append("slim/")

from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory
slim = tf.contrib.slim


class SlimModelEncoder(object):
  def __init__(self, name, num_classes, is_training):
    utils.slim_maybe_download(name)
    self.model_name = name
    self.network_fn = nets_factory.get_network_fn(name, num_classes, is_training=is_training)
    self.network_arg_scope = nets_factory.arg_scopes_map[name]
    self.preprocessing_fn = preprocessing_factory.get_preprocessing(self.model_name)

  def build(self, image):
    tf.logging.set_verbosity(tf.logging.INFO)
    # preprocess images. the image might need to be reshaped to cater to the model used.
    h = w = self.network_fn.default_image_size
    # TODO: This takes one image at a time :(
    # get the next batch from the dataset iterator
    processed_images = self.preprocessing_fn(image, h, w)
    processed_images = tf.expand_dims(processed_images, 0)


    # build the model with the arg scopes - common params
    with slim.arg_scope(self.network_arg_scope()):
      _, end_points = self.network_fn(processed_images)

    # load the variables
    _model_ckpt_name = self.model_name + '.ckpt'
    init_fn = slim.assign_from_checkpoint_fn(
      os.path.join(utils.CHECKPOINTS_DIR, _model_ckpt_name),
      slim.get_model_variables(self.model_name))
    return init_fn, end_points


class FCNDecoder(object):
  def __init__(self, end_points, nb_classes, scope):
    "tensors to connect shd include the output tensor"
    self.end_points = end_points
    self.nb_classes = nb_classes
    self.scope = scope

  def convolve(self, layer):
    return slim.conv2d(layer, self.nb_classes, [1, 1], padding='same')

  def upsample(self, layer, stride):
    #TODO: resize_image initialization instead.
    return slim.conv2d_transpose(layer, self.nb_classes, kernel_size=4, stride=stride, padding='same')

  def build(self, tensors_to_connect):
    with tf.variable_scope(self.scope, 'bla', [list(self.end_points.values())]) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d],
                          weights_regularizer=slim.l2_regularizer(1e-4),
                          outputs_collections=end_points_collection):
        for i, (layer_name, stride) in enumerate(tensors_to_connect.items()):
          layer = self.end_points[layer_name]
          layer = self.convolve(layer)
          if i > 0:
            net = tf.add(net, layer)
            net = self.upsample(net, stride)
          else:
            net = self.upsample(layer, stride)
    return net
















#
# """
# Load an encoder saved as `saved_model`.
# """
# class SavedModelEncoder(object):
# 	def __init__(self, name, saved_model_path):
# 		if tf.saved_model.loader.maybe_saved_model_directory(saved_model_path):
# 			tf.saved_model.loader.load()
#
#
#
# 	def build(self, inputs):
# 		init_fn, end_points = None, None
#
# 		return init_fn, end_points
#
#
#
#
#
# class Decoder(object):
# 	def __init__(self):
# 		self.variables = None
# 		self.graph_def = None
