import collections
import tensorflow as tf


slim = tf.contrib.slim



class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a DenseNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The DenseNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the DenseNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """

@slim.add_arg_scope
def stack_blocks_dense(net,
                       growth_rate,
                       bottleneck_number_feature_maps=None,
                       dropout_keep_prob=0.2,
                       **kwargs):
  """Stacks DenseNet units"""
  # bottleneck if defined
  if bottleneck_number_feature_maps:
    net = slim.batch_norm(net,
                        activation_fn=tf.nn.relu,
                        scope='batch_norm_0')
    net = slim.conv2d(net, bottleneck_number_feature_maps,
                    kernel_size=1,
                    normalizer_fn=None,
                    activation_fn=None,
                    scope='bottleneck')
    if dropout_keep_prob:
      net = slim.dropout(net,
	                   keep_prob=dropout_keep_prob,
	                   scope='dropout_0')
  # convolution
  net = slim.batch_norm(net,
                        activation_fn=tf.nn.relu,
                        scope='batch_norm_1')
  net = slim.conv2d(net, growth_rate,
                    kernel_size=3,
                    normalizer_fn=None,
                    activation_fn=None,
                    scope='conv')

  if dropout_keep_prob:
    net = slim.dropout(net,
	                     keep_prob=dropout_keep_prob,
	                     scope='dropout_1')
  return net

@slim.add_arg_scope
def add_transition_down_layer(net,
                              dropout_keep_prob=0.2,
                              compression_factor=1.0):
  depth_in = slim.utils.last_dimension(net.get_shape(), min_rank=4)
  net = slim.batch_norm(net,
                        activation_fn=tf.nn.relu,
                        scope='batch_norm')
  net = slim.conv2d(net, depth_in,
                    kernel_size=1,
                    normalizer_fn=None,
                    activation_fn=None,
                    scope='transition')
  if dropout_keep_prob:
    net = slim.dropout(net,
	                     keep_prob=dropout_keep_prob,
	                     scope='dropout_1')
  return slim.avg_pool2d(net, [2, 2], padding='same',scope='pool')

@slim.add_arg_scope
def add_transition_up_layer(net,
                            stride=2,
                            kernel_size=3,
                            activation=None,
                            scope=None):
  # no activation between layers in upsampling
  depth_in = slim.utils.last_dimension(net.get_shape(), min_rank=4)
  return slim.conv2d_transpose(net,
                               depth_in,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding='same',
                               activation_fn=activation,
                               scope=scope)










