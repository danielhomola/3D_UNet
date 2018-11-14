"""
3D U-Net network, for prostate MRI scans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.network import unet_3d_network

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
    """
    Custom estimator setup as per docs and guide:
    https://www.tensorflow.org/guide/custom_estimators

    Several ideas taken from: https://github.com/cs230-stanford/
    cs230-code-examples/tree/master/tensorflow/vision

    Args:
        features: This is batch_features from input_fn.
        labels: This is batch_labels from input_fn.
        mode (:class:`tf.estimator.ModeKeys`): Train, eval, or predict.
        params (dict): Params for setting up the model. Expected keys are:
            feature_columns (list <:class:`tf.feature_column`>): Feature types.
            depth (int): Depth of the architecture.
            n_base_filters (int): Number of conv3d filters in the first layer.
            num_classes (int): Number of mutually exclusive output classes.
            class_weights (:class:`numpy.array`): Weight of each class to use.
            learning_rate (float): LR to use with Adam.
            batch_norm (bool): Whether to use batch_norm in the conv3d blocks.

    Returns:
        :class:`tf.estimator.Estimator`: A 3D U-Net network, as TF Estimator.
    """

    # -------------------------------------------------------------------------
    # get logits from 3D U-Net
    # -------------------------------------------------------------------------

    training = mode == tf.estimator.ModeKeys.TRAIN
    logits = unet_3d_network(inputs=features, params=params, training=training)

    # -------------------------------------------------------------------------
    # predictions - for PREDICT and EVAL modes
    # -------------------------------------------------------------------------

    prediction = tf.argmax(logits, axis=-1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': prediction,
            'probabilities': tf.nn.softmax(logits, axis=-1)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # -------------------------------------------------------------------------
    # loss - for TRAIN and EVAL modes
    # -------------------------------------------------------------------------

    # weighted softmax, see https://stackoverflow.com/a/44563055
    class_weights = tf.cast(tf.constant(params['class_weights']), tf.float32)
    class_weights = tf.reduce_sum(
        tf.cast(features, tf.float32) * class_weights, axis=-1
    )
    loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        weights=class_weights
    )

    # -------------------------------------------------------------------------
    # metrics: mean IOU - for TRAIN and EVAL modes
    # -------------------------------------------------------------------------

    labels_dense = tf.argmax(labels, -1)
    iou = tf.metrics.mean_iou(
        labels=labels_dense,
        predictions=tf.cast(prediction, tf.int32),
        num_classes=params['num_classes'],
        name='iou_op'
    )
    metrics = {'iou': iou}

    # save metric both for logging and for tensorboard
    tf.identity(iou[0], name='train_iou')
    tf.summary.scalar('iou', iou[0])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # -------------------------------------------------------------------------
    # train op - for TRAIN
    # -------------------------------------------------------------------------

    assert mode == tf.estimator.ModeKeys.TRAIN

    # as per TF batch_norm docs and also following https://goo.gl/1UVeYK

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    global_step = tf.train.get_or_create_global_step()
    if params['batch_norm']:
        # Add dependency to update the moving mean and variance of batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
