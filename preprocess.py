import os
import sys

import tensorflow as tf
import utils
import densenet_utils


sys.path.append("slim/")

from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory
slim = tf.contrib.slim

def preprocess(image, scope):
	with tf.variable_scope(scope, values=[image]):
		# dynamic shape
		image_shape = tf.shape(image)[:2]
		image_shape = image_shape - tf.floormod(image_shape, 32)
		image_shape = tf.cast(image_shape, tf.int32)
		#tf.Print(image_shape, [image_shape])
		# image_shape = tf.Print(image_shape, [image_shape])
		# TODO: image normalization?!?
		# get the next batch from the dataset iterator
		processed_images = tf.image.resize_images(image, image_shape)
		processed_images = tf.expand_dims(processed_images, 0)
		print("preprocessing tensor output {}".format(processed_images))
		return processed_images