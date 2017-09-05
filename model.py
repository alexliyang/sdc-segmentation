
import os
import sys

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
