
import tensorflow as tf
import os
import logging

from slim.datasets import dataset_utils

CHECKPOINTS_DIR = 'model_checkpoints/'

PRETRAINED_MODEL_PATHS = \
	{"vgg_16": "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"}

VARIABLES_TO_EXCLUDE = \
	{"vgg_16": "fc8"}

def slim_maybe_download(name):
	if not tf.gfile.Exists(CHECKPOINTS_DIR):
		tf.gfile.MakeDirs(CHECKPOINTS_DIR)
	model_checkpoint_path = os.path.join(CHECKPOINTS_DIR, name + ".ckpt")
	if not os.path.exists(model_checkpoint_path):
		url = PRETRAINED_MODEL_PATHS[name]
		dataset_utils.download_and_uncompress_tarball(url, CHECKPOINTS_DIR)
		logging.info("download successful")


def saved_model_maybe_download(name, url):
	if not tf.gfile.Exists(CHECKPOINTS_DIR):
		tf.gfile.MakeDirs(CHECKPOINTS_DIR)
	saved_model_path = os.path.join(CHECKPOINTS_DIR, name + ".pb")
	if not os.path.exists(saved_model_path):
		dataset_utils.download_and_uncompress_tarball(url, CHECKPOINTS_DIR)
		logging.info("download successful")