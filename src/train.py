"""
Train 3D U-Net network, for prostate MRI scans.

Ideas taken from:
https://github.com/cs230-stanford/cs230-code-examples/tree/master/
tensorflow/vision

and

https://github.com/tensorflow/models/blob/master/samples/core/
get_started/custom_estimator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf

from src.model_fn import model_fn
from src.input_fn import input_fn
from src.utils import Params, set_logger

# setup command line args
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")


def main(argv):
    # set the random seed for the whole graph for reproducible experiments
    tf.set_random_seed(230)

    # load the parameters from model's json file
    args = parser.parse_args(argv[1:])
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict

    # set the logger, add IOU to logging
    set_logger(os.path.join(args.model_dir, 'train.log'))

    tensors_to_log = {"Mean IOU": "iou"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # create and train model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=params
    )

    model.train(
        input_fn=lambda: input_fn(True, params),
        steps=params['train_steps'],
        hooks=[logging_hook])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)