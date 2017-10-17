from __future__ import print_function

from input import *
from models import *
from train import *

NB_CLASSES = 151

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  iterator, filename = get_train_inputs(batch_size=1,
	                                      repeat=True,
	                                      num_classes=1)

  # Model
  image, label = iterator.get_next()
  image = tf.to_float(image)
  encoder = DeepLabV3(name='resnet_v2_50',
                      num_classes=NB_CLASSES,
                      is_training=True)
  restore_fn, net = encoder.build(image=image,
                                  global_pool=False,
                                  output_stride=16)
  # Train
  trainer = Trainer(nb_classes=NB_CLASSES,
                    optimizer=tf.train.AdamOptimizer, learning_rate=1e-4)
  trainer.build(predictions=net, labels=label, one_hot=True)
  trainer.train(iterator,
                restore_fn=restore_fn,
                number_of_steps=10000,
                filename=['data/adek20_training.tfecord'])

if __name__ == '__main__':
  tf.app.run()