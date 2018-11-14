"""
Data feeding function for train and test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
            train_img_size (int): Width and height of resized training images.
            batch_size (int): Number of of patient in each batch for training.
            num_classes (int): Number of mutually exclusive output classes.
            train_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.
            test_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.

    Returns:
        :class:`tf.dataset.Dataset`: An instantiated Dataset object.
    """
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # for training we use a batch number and pad each 3D scan to have equal
    # depth, width and height have already been set to 128 in preprocessing
    max_s = params['max_scans']
    w = h = params['train_img_size']
    if training:
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['train_dataset_path'])
        ).create_tf_dataset().shuffle(
            # we have 70 train examples, this will provide good shuffling
            buffer_size=70 
        ).repeat().padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
            [max_s, w, h, 1], [max_s, w, h, params['num_classes']])
        )

    # for testing we use the unscaled images with their original dims,
    # we still pad the depth dimension to max_s though
    else:
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['test_dataset_path'])
        ).create_tf_dataset().padded_batch(
            batch_size=1,
            padded_shapes=(
            [max_s, None, None, 1], [max_s, None, None, params['num_classes']])
        )

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init_op = iterator.make_initializer(dataset)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init_op)
    next_element = iterator.get_next()
    return next_element[0], next_element[1]
