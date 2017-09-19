from __future__ import print_function
from collections import OrderedDict

from input import *
from model import *
from train import *

NB_CLASSES = 151



def main(_):
	# Input
  iterator, filename = get_train_inputs(batch_size=1,
	                                      repeat=True,
	                                      num_classes=1)

  encoder = SlimModelEncoder(name="vgg_16", num_classes=NB_CLASSES, is_training=True)
  image, label = iterator.get_next()
  image = tf.to_float(image)
  restore_fn, end_points = encoder.build(image=image)
  decoder = FCNDecoder(end_points, nb_classes=NB_CLASSES, scope='decoder')

  tensors_to_connect = OrderedDict()
  tensors_to_connect["vgg_16/fc8"] = (2,2)
  tensors_to_connect['vgg_16/pool4'] = (2,2)
  tensors_to_connect['vgg_16/pool3'] = (8,8)
  net = decoder.build(tensors_to_connect)

  # Train
  trainer = Trainer(nb_classes=NB_CLASSES, optimizer=tf.train.AdamOptimizer, learning_rate=1e-4)
  trainer.build(predictions=net, labels=label)
  trainer.train(iterator,
	              restore_fn=restore_fn,
	              number_of_steps=2500,
	              filename=['data/adek20_training.tfecord'])

if __name__ == '__main__':
  tf.app.run()








