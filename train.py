import sys

import tensorflow as tf

sys.path.append("slim/")

slim = tf.contrib.slim

TRAIN_DIR = "/tmp"


class Trainer(object):
  def __init__(self, nb_classes, optimizer, learning_rate):
    self.nb_clasess = nb_classes
    # learning rate can be a placeholder tensor
    self.learning_rate = learning_rate
    self.optimizer = optimizer(learning_rate)
    self.train_op = None

  def build(self, predictions, labels):
    predictions = tf.reshape(predictions, (-1, self.nb_clasess))
    labels = tf.reshape(labels, (-1, self.nb_clasess))
    # wraps the softmax_with_entropy fn. adds it to loss collection
    tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=labels)
    # include the regulization losses in loss collection.
    total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
    # train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too.
    self.train_op = slim.learning.create_train_op(total_loss, self.optimizer)

  def train(self, number_of_steps=1000, same_summaries_secs=300, save_interval_secs=600):
    slim.learning.train(train_op=self.train_op,
                        logdir=TRAIN_DIR,
                        number_of_steps=number_of_steps,
                        save_summaries_secs=same_summaries_secs,
                        save_interval_secs=save_interval_secs)



