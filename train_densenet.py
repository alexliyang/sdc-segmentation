from __future__ import print_function
from collections import OrderedDict

from input import *
from model import *
from train import *

NB_CLASSES = 151



def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	iterator, filename = get_train_inputs(batch_size=1,
	                                      repeat=True,
	                                      num_classes=1)
	image, label = iterator.get_next()
	image = tf.to_float(image)

	# encoder
	densenet = DenseNet(growth_rate=12, global_pool=False, spatial_squeeze=False)
	num_units = [6, 12 ,24, 16]
	bottleneck_number_feature_maps = 12 * 4
	net, end_points = densenet.build(image=image,
	                                        scope='encoder',
	                                        num_units=num_units,
	                                        bottleneck_number_feature_maps=bottleneck_number_feature_maps)
	# decoder
	skip_connection_collection = 'skip_connections'
	num_units = [12, 10, 10]
	encoder = Tiramisu(skip_connection_collection=skip_connection_collection,
	                   num_classes=NB_CLASSES,
	                   num_units=num_units)
	net, decoder_end_points = encoder.build(net, "decoder", 12)
	# Train
	trainer = Trainer(nb_classes=NB_CLASSES, optimizer=tf.train.AdamOptimizer, learning_rate=1e-4)
	trainer.build(predictions=net, labels=label, one_hot=True)
	trainer.train(iterator,
	              number_of_steps=2500,
	              filename=['data/adek20_training.tfecord'])

if __name__ == '__main__':
	tf.app.run()

