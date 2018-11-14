"""
Data feeding function for train and test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from src.data_utils import Dataset


def input_fn(training, params):
    """
    Simple input_fn for our 3D U-Net estimator, handling train and test data
    preparation.

    Args:
        training (bool): Whether we are training or testing.
        params (dict): Params for setting up the data. Expected keys are:
            max_scans (int): Maximum number of scans we see in any patient.
            img_size (int): Width and height of the resized training images.
            batch_size (int): Number of of patient in each batch for training.
            num_classes (int): Number of mutually exclusive output classes.

    Returns:
        :class:`tf.dataset.Dataset`: An instantiated Dataset object.
    """

    # for training we use a batch number and pad each 3D scan to have equal
    # depth, width and height have already been set to 128 in preprocessing
    if training:
        dataset = Dataset.load_dataset('../data/processed/train_dataset.pckl')
        max_s = params['max_scans']
        w = h = params['img_size']

        dataset = dataset.create_tf_dataset().shuffle(
            70
        ).repeat().padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
            [max_s, w, h, 1], [max_s, w, h, params['num_classes']])
        )

    #
    else:
        dataset = Dataset.load_dataset('../data/processed/test_dataset.pckl')
        dataset.create_tf_dataset().batch(1)

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init_op = iterator.make_initializer(dataset)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init_op)
    next_element = iterator.get_next()
    return next_element[0], next_element[1]
