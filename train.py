import sys

import tensorflow as tf
from tensorflow.python.ops import math_ops

sys.path.append("slim/")

slim = tf.contrib.slim

TRAIN_DIR = "/tmp/tf"


class Trainer(object):
  def __init__(self, nb_classes, optimizer, learning_rate):
    self.nb_clasess = nb_classes
    # learning rate can be a placeholder tensor
    self.learning_rate = learning_rate
    self.optimizer = optimizer(learning_rate)
    self.train_op = None

  def build(self, predictions, labels, decoder_scope):
    predictions = tf.reshape(predictions, (-1, self.nb_clasess))
    labels = tf.expand_dims(labels, 0)
    labels = tf.reshape(labels, (-1, self.nb_clasess))
    print("pred shape {}, label shape {}".format(predictions.get_shape(), labels.get_shape()))

    # wraps the softmax_with_entropy fn. adds it to loss collection
    tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=labels)
    # include the regulization losses in the loss collection.
    # take all regulization losses
    #reg_losses = tf.losses.get_regularization_losses(scope=decoder_scope)
    reg_losses = tf.losses.get_regularization_losses()
    print("reg losses: {}".format(reg_losses))
    loss = tf.losses.get_losses()
    loss += reg_losses
    total_loss = math_ops.add_n(loss, name='total_loss')
    # train_op ensures that each time we ask for the loss,
    # the gradients are computed and applied.
    #variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, decoder_scope)
    # train end to end
    # self.train_op = slim.learning.create_train_op(total_loss,
    #                                               optimizer=self.optimizer,
    #                                               variables_to_train=variables_to_train)
    self.train_op = slim.learning.create_train_op(total_loss,
                                                  optimizer=self.optimizer)

  def train(self, iterator,
            assign_op,
            feed_dict,
            filename,
            number_of_steps=10000,
            same_summaries_secs=120,
            keep_checkpoint_every_n_hours=0.5):
    # Add summaries for variables and losses.
    global_summaries = set([])
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    # total loss
    total_loss_tensor = tf.get_default_graph().get_tensor_by_name('total_loss:0')
    global_summaries.add(tf.summary.scalar(total_loss_tensor.op.name, total_loss_tensor))
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(global_summaries), name='summary_op')
    # Save checkpoints regularly.
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
    # init fn for the dataset ops and checkpointin
    def initializer_fn(sess):
        input_tensor = tf.get_default_graph().get_tensor_by_name('training_data/input:0')
        sess.run(iterator.initializer, feed_dict={input_tensor: filename})
        #tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        sess.run(assign_op, feed_dict=feed_dict)
    init_fn = initializer_fn
    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True)
    # train
    slim.learning.train(train_op=self.train_op,
                        logdir=TRAIN_DIR,
                        session_config=session_config,
                        summary_op=summary_op,
                        init_fn=init_fn,
                        number_of_steps=number_of_steps,
                        save_summaries_secs=same_summaries_secs,
                        saver=saver)
