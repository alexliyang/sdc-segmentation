from __future__ import print_function
from collections import OrderedDict

from input import *
from model import *
from train import *

NB_CLASSES = 151



def main(_):
	iterator, filename = get_train_inputs(batch_size=1,
	                                      repeat=True,
	                                      num_classes=1)
	image, label = iterator.get_next()
	image = tf.to_float(image)
	# TODO: Preprocess

	densenet = DenseNet(growth_rate=12, global_pool=False, spatial_squeeze=False)
	num_units = [6, 12 , 24, 16]
	bottleneck_number_feature_maps = 12 * 4
	restore_fn, end_points = densenet.build(image=image,
	                                        scope='encoder',
	                                        num_units=num_units,
	                                        bottleneck_number_feature_maps=bottleneck_number_feature_maps)

	print(end_points)


if __name__ == '__main__':
	tf.app.run()

