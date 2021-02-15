import argparse
import glob
import os
import shutil
import random
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function

    # Number of tfrecords
    train_size = 75
    val_size = 15
    test_size = 10

    filenames = [filename for filename in os.listdir(
        data_dir+"/processed/") if filename.endswith(".tfrecord")]
    random.shuffle(filenames)
    os.makedirs(data_dir+"/train", exist_ok=True)
    os.makedirs(data_dir+"/val", exist_ok=True)
    os.makedirs(data_dir+"/test", exist_ok=True)

    for i, filename in enumerate(filenames):
        if i < train_size:
            directory = 'train'
        elif i < train_size + val_size:
            directory = 'val'
        elif i <= train_size + val_size + test_size:
            directory = 'test'

        shutil.move(data_dir+"/processed/"+filename,
                    data_dir+"/"+directory+"/"+filename)

    df = pd.DataFrame({'train[tfrecords]': [train_size], 'test[tfrecords]': [test_size], 'val[tfrecords]': [val_size]},
                      index=['size'])
    print(df.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
