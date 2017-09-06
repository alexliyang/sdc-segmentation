
import os
import sys
import numpy as np

import tensorflow as tf
import utils


sys.path.append("slim/")

from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory
slim = tf.contrib.slim


"""
Use Slim Pre-trained models as encoder
"""
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
	def __init__(self, tensors_to_connect, input_tensor, nb_classes, scope):
		self.tensors_to_connect = tensors_to_connect
		self.input_tensor = input_tensor
		self.nb_classes = nb_classes
		self.scope = scope

	# def upsample(layer, nb_classes):
	# 	output = tf.layers.conv2d_transpose(layer, nb_classes, kernel_size=4, strides=(2, 2),
	# 	                                    padding='same',
	# 	                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
	# 		return output

	def _compute_strides(self, end_points):
		nb_upsample_filters = len(self.tensors_to_connect)
		map_size_w = np.zeros(nb_upsample_filters)
		map_size_h = np.zeros(nb_upsample_filters)
		stride_w = np.zeros(nb_upsample_filters)
		stride_h = np.zeros(nb_upsample_filters)
		for i in range(nb_upsample_filters):
			map_size_h[i] = end_points[self.tensors_to_connect[i]].get_shape().as_list()[2]
			map_size_w[i] = end_points[self.tensors_to_connect[i]].get_shape().as_list()[3]
			stride_h[i] = map_size_h[i] / map_size_h[i-1] if i > 0 else None
			stride_w[i] = stride_w[i] / stride_w[i - 1] if i > 0 else None
		stride_w = stride_w[~np.isnan(stride_w)]
		stride_h = stride_h[~np.isnan(stride_h)]
		return stride_w, stride_h

	# def get_deconv_filter(self, f_shape):
	# 	# bilinear upsampling initializer
	# 	width = f_shape[0]
	# 	heigh = f_shape[0]
	# 	f = ceil(width / 2.0)
	# 	c = (2 * f - 1 - f % 2) / (2.0 * f)
	# 	bilinear = np.zeros([f_shape[0], f_shape[1]])
	# 	for x in range(width):
	# 		for y in range(heigh):
	# 			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
	# 			bilinear[x, y] = value
	# 	weights = np.zeros(f_shape)
	# 	for i in range(f_shape[2]):
	# 		weights[:, :, i, i] = bilinear
	#
	# 	init = tf.constant_initializer(value=weights,
	# 	                               dtype=tf.float32)
	# 	var = tf.get_variable(name="up_filter", initializer=init,
	# 	                      shape=weights.shape)
	# 	return var

	def build(self, end_points):
		stride_w, stride_h = self._compute_strides(end_points)

		with tf.variable_scope(self.scope, 'fcn', [list(end_points.values())]) as sc:
			end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
			                    weights_regularizer=slim.l2_regularizer(1e-4),
			                    weights_initializer=slim.l2_regularizer(1e-4),
			                    outputs_collections=end_points_collection):
			for i in range(self.nb_upsample_filters-1):
				slim.conv2d(end_points[self.tensors_to_connect[i]],
				            self.nb_classes,
				            [stride_h, stride_w],
				            padding='same')
















"""
Load an encoder saved as `saved_model`.
"""
class SavedModelEncoder(object):
	def __init__(self, name, saved_model_path):
		if tf.saved_model.loader.maybe_saved_model_directory(saved_model_path):
			tf.saved_model.loader.load()



	def build(self, inputs):
		init_fn, end_points = None, None

		return init_fn, end_points





class Decoder(object):
	def __init__(self):
		self.variables = None
		self.graph_def = None
