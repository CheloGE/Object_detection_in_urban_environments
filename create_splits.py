import argparse
import glob
import os
import random
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from utils import get_module_logger


def tfDataset2tfrecords(dataset, nameOfFolder):
    record_every = 200
    i = 0
    for i, _ in dataset.enumerate():
        i = i.numpy()
        if i % record_every == 0:
            if i == 0:
                temp_ds = dataset
            else:
                temp_ds = temp_ds.skip(record_every)
            recordNum = int(i/record_every+1)
            print(
                f"\rwritting new tfrecord {nameOfFolder} file... {recordNum}", end='', flush=True)
            writer = tf.data.experimental.TFRecordWriter(
                f"../data/processed/{nameOfFolder}/{nameOfFolder}-{recordNum}.tfrecord")
            writer.write(temp_ds.take(record_every))
    print("")
    return i


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function

    filenames = [
        data_dir+filename for filename in os.listdir(data_dir) if filename.endswith(".tfrecord")]
    os.makedirs(data_dir+"/train", exist_ok=True)
    os.makedirs(data_dir+"/val", exist_ok=True)
    os.makedirs(data_dir+"/test", exist_ok=True)
    DATASET_SIZE = 19802  # This number is taken from EDA analysis
    train_size = int(0.65 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    full_dataset = tf.data.TFRecordDataset(filenames)
    # we need to set reshuffle to False because we don't want different data on each take, skip steps
    full_dataset = full_dataset.shuffle(1000, reshuffle_each_iteration=False)
    full_dataset = full_dataset.repeat(1)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_size = tfDataset2tfrecords(train_dataset, 'train')
    test_size = tfDataset2tfrecords(test_dataset, 'test')
    val_size = tfDataset2tfrecords(val_dataset, 'val')
    # we erase all tfrecords from processed folder
    os.system(f'rm -rf {data_dir}/*.tfrecord')

    df = pd.DataFrame({'train': [train_size], 'test': [test_size], 'val': [val_size]},
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
