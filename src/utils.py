"""
General utility functions. Taken from: https://github.com/cs230-stanford/
cs230-code-examples/tree/master/tensorflow/vision
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging


class Params():
    """
    Class that loads hyperparameters from a json file.
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """
        Saves parameters to json file
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to Params instance by
        `params.dict['learning_rate']`
        """
        return self.__dict__


def set_logger(log_path):
    """
    Sets the logger to log info in terminal and file `log_path`.
    Log file saved to `model_dir/train.log`.


    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)
    
    # Log to model folder to a file to not just to console
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    )
    logger.addHandler(file_handler)
    return logger
