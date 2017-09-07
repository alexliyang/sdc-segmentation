import os
import sys

import tensorflow as tf

sys.path.append("slim/")

slim = tf.contrib.slim


class Trainer(object):
  def __init__(self, nb_classes, learning_rate):
    self.nb_clasess = nb_classes
    # learning rate can be a placeholder tensor
    self.learning_rate = learning_rate

  def build(self, predictions, labels):
    predictions = tf.reshape(predictions, (-1, self.nb_clasess))
    labels = tf.reshape(labels, (-1, self.nb_clasess))
    # wraps the softmax_with_entropy fn. adds it to loss collection
    tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=labels)
    # include the regulization losses in loss collection.
    total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
    # optimizier
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too.
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    return train_op, optimizer

  def train(self):
    pass



