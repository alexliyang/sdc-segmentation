
import os
import sys

import tensorflow as tf
from tensorflow.python.ops import nn
import utils
import densenet_utils


sys.path.append("slim/")

from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory
slim = tf.contrib.slim


class SlimModelEncoder(object):
  def __init__(self, name, num_classes, is_training):
    utils.slim_maybe_download(name)
    self.model_name = name
    self.variables_to_exclude = utils.VARIABLES_TO_EXCLUDE[name]
    # need to set weight decay here because it gets called after the arg_scope
    self.network_fn = nets_factory.get_network_fn(name, num_classes, is_training=is_training, weight_decay=0.0005)
    self.network_arg_scope = nets_factory.arg_scopes_map[name]
    self.preprocessing_fn = preprocessing_factory.get_preprocessing(self.model_name,
                                                                    is_training=is_training)

  def build(self, image):
    tf.logging.set_verbosity(tf.logging.INFO)
    # TODO: This takes one image at a time :(
    # preprocess images. the image might need to be reshaped to cater to the model used.
    image_shape = tf.shape(image)[:2]
    image_shape = image_shape - tf.floormod(image_shape, 32)
    image_shape = tf.cast(image_shape, tf.int32)
    #image_shape = tf.Print(image_shape, [image_shape])
    print(image_shape)
    # TODO: Where is preproessing image normalization?
    # get the next batch from the dataset iterator
    processed_images = tf.image.resize_images(image, image_shape)
    processed_images = tf.expand_dims(processed_images, 0)

    # build the model with the arg scopes - common params
    with slim.arg_scope(self.network_arg_scope()):
      _, end_points = self.network_fn(processed_images,
                                      fc_conv_padding='same',
                                      spatial_squeeze=False,
                                      dropout_keep_prob=1.0)
    # create an op to assign variables from a checkpoint
    _model_ckpt_name = self.model_name + '.ckpt'
    _var_list = slim.get_variables(self.model_name)
    _filtered_var_list = slim.filter_variables(_var_list, exclude_patterns=[self.variables_to_exclude])
    print("restore following variables: {}".format(_filtered_var_list))
    restore_op = slim.assign_from_checkpoint_fn(
      os.path.join(utils.CHECKPOINTS_DIR, _model_ckpt_name),
      _filtered_var_list
      )
    return restore_op, end_points


class FCNDecoder(object):
  def __init__(self, end_points, nb_classes, scope):
    "tensors to connect shd include the output tensor"
    self.end_points = end_points
    self.nb_classes = nb_classes
    self.scope = scope

  def convolve(self, layer, activation=None,scope=None):
    # no activation for 1x1 convolution
    return slim.conv2d(layer, self.nb_classes, 1, padding='same',
                       activation_fn=activation,
                       scope=scope)

  def upsample(self, layer, stride, kernel_size, activation=None, scope=None):
    # no activation between layers
    return slim.conv2d_transpose(layer, self.nb_classes, kernel_size=kernel_size,
                                 stride=stride,
                                 padding='same',
                                 activation_fn=activation,
                                 scope=scope)

  def build(self, tensors_to_connect):
    with tf.variable_scope(self.scope, values=tensors_to_connect) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d,slim.conv2d_transpose],
                          weights_regularizer=None,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          outputs_collections=end_points_collection):
        scope = 'upsample_conv_'
        for i, (layer_name, stride) in enumerate(tensors_to_connect.items()):
          layer = self.end_points[layer_name]
          layer = self.convolve(layer)
          if i > 0:
            net = tf.add(net, layer)
            if stride == (8, 8):
              # use a larger kernel for the last upsampling layer
              net = self.upsample(net, stride, kernel_size=16, scope=scope+str(i))
            else:
              net = self.upsample(net, stride, kernel_size=4, scope=scope+str(i))
          else:
            net = self.upsample(layer, stride, kernel_size=4, scope=scope+str(i))
    return net


class DenseNet(object):
  def __init__(self,
               growth_rate,
               is_training=True,
               reuse=None,
               num_classes=None):
    self.growth_rate = growth_rate
    self.reuse = reuse
    self.is_training = is_training

  def build(self, inputs,
            scope,
            num_units,
            bottleneck_number_feature_maps=None,
            dropout_keep_prob=0.2):
    """
    Args:
    	inputs: image.
      scope: The scope of the block.
      bottleneck_number_feature_maps: If not None, 
        1x1 bottleneck reduces the output
      dropout_rate: dropout rate to apply for each conv unit in training.
      num_units: Array of number of units in each block.
    """
    rate = self.growth_rate
    bottleneck_maps = bottleneck_number_feature_maps

    with tf.variable_scope(scope, 'densenet', [inputs], reuse=self.reuse) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d,
                           densenet_utils.stack_blocks_dense],
                          # don't use bias for any convolution
                          biases_initializer = False,
                          outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=self.is_training):
          net = inputs
          # TODO: Normalize the input
          initial_nb_layers = self.growth_rate * 2
          net = slim.conv2d(net, initial_nb_layers, kernel_size=7,
                            padding='same',
                            stride=2,
                            scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

          for bn, num_units_in_block in enumerate(num_units):
            with tf.variable_scope("block_".format(bn + 1), values=[net]) as sc:
              for un, unit in enumerate(range(num_units_in_block)):
                with tf.variable_scope("unit_".format(un + 1), values=[net]):
                  output = densenet_utils.stack_blocks_dense(net,
                                                             growth_rate=rate,
                                                             bottleneck_number_feature_maps=bottleneck_maps)
                  net = tf.concat(axis=3, values=[net, output])
              # the last layer does not have a transition layer
              if bn + 1 != len(num_units):
                densenet_utils.add_transition_layer()











class Tiramisu(object):
  def __init(self):
    pass
















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
