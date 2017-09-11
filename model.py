
import os
import sys

import tensorflow as tf
from tensorflow.python.ops import nn
import utils


sys.path.append("slim/")

from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory
slim = tf.contrib.slim


class SlimModelEncoder(object):
  def __init__(self, name, num_classes, is_training):
    utils.slim_maybe_download(name)
    self.model_name = name
    self.variables_to_exclude = utils.VARIABLES_TO_EXCLUDE[name]
    # TODO: Do I even need to pass `num_classes_ here?
    self.network_fn = nets_factory.get_network_fn(name, num_classes, is_training=is_training)
    self.network_arg_scope = nets_factory.arg_scopes_map[name]
    self.preprocessing_fn = preprocessing_factory.get_preprocessing(self.model_name,
                                                                    is_training=is_training)

  def _get_variables_to_train(self, scopes):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    variables_to_train = []
    for scope in scopes:
      variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
      variables_to_train.extend(variables)
    return variables_to_train

  def build(self, image, image_shape):
    tf.logging.set_verbosity(tf.logging.INFO)
    # preprocess images. the image might need to be reshaped to cater to the model used.
    #h = w = self.network_fn.default_image_size
    h, w = image_shape
    # TODO: This takes one image at a time :(
    # get the next batch from the dataset iterator
    processed_images = self.preprocessing_fn(image, h, w)
    processed_images = tf.expand_dims(processed_images, 0)

    # build the model with the arg scopes - common params
    with slim.arg_scope(self.network_arg_scope()):
      _, end_points = self.network_fn(processed_images,
                                      fc_conv_padding='same',
                                      spatial_squeeze=False)
    # create an op to assign variables from a checkpoint
    _model_ckpt_name = self.model_name + '.ckpt'
    _var_list = slim.get_variables(self.model_name)
    _filtered_var_list = slim.filter_variables(_var_list, exclude_patterns=self.variables_to_exclude)
    assign_op, feed_dict = slim.assign_from_checkpoint(
      os.path.join(utils.CHECKPOINTS_DIR, _model_ckpt_name),
      _filtered_var_list
      )
    return assign_op, feed_dict, end_points


class FCNDecoder(object):
  def __init__(self, end_points, nb_classes, scope):
    "tensors to connect shd include the output tensor"
    self.end_points = end_points
    self.nb_classes = nb_classes
    self.scope = scope

  def convolve(self, layer):
    return slim.conv2d(layer, self.nb_classes, [1, 1], padding='same')

  def upsample(self, layer, stride, activation=None, kernel_size=4):
    return slim.conv2d_transpose(layer, self.nb_classes, kernel_size=kernel_size,
                                 stride=stride,
                                 padding='same',
                                 activation_fn=activation)

  def build(self, tensors_to_connect):
    with tf.variable_scope(self.scope, values=tensors_to_connect) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d],
                          weights_regularizer=slim.l2_regularizer(1e-4),
                          outputs_collections=end_points_collection):
        for i, (layer_name, stride) in enumerate(tensors_to_connect.items()):
          layer = self.end_points[layer_name]
          layer = self.convolve(layer)
          if i > 0:
            net = tf.add(net, layer)
            if stride == (4,4):
              # use a larger kernel for the last upsampling layer
              net = self.upsample(net, stride, kernel_size= 16)
            else:
              net = self.upsample(net, stride, kernel_size= 4)
          else:
            net = self.upsample(layer, stride)
    net = tf.identity(net, name="logit")
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
