# -*- coding: utf-8 -*-
from tkinter import W
import click
import logging
import os, wget, tarfile
import numpy as np
import pandas as pd
import pickle
import random
import re
import matplotlib.pyplot as plt
import re
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def download_data(destination_dir, url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"):

    # Make path if does not exits
    if(os.path.exists(destination_dir) == False):
        os.makedirs(destination_dir)
    # Download data if it does not exist on disk
    if(os.path.exists(os.path.join(destination_dir, "cifar-10-python.tar.gz")) == False):
        print("\n")
        wget.download(url, destination_dir)
        print("\n")

def extract_data(data_dir, filename="cifar-10-python.tar.gz"):

    file_handler = tarfile.open(os.path.join(data_dir, filename))
    file_handler.extractall(data_dir)
    file_handler.close()

def unpickle(file):

    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def load_train_data(data_dir):

    train_filenames = []
    temp_data = []
    temp_labels = []

    for filename in os.listdir(data_dir):
        if re.search("^data_batch_[0-9]$", filename):
            train_filenames.append(filename)

    for filename in train_filenames:
        temp_data_path = os.path.join(data_dir, filename)
        temp_data.append(unpickle(temp_data_path)[b'data'])
        temp_labels.append(np.array(unpickle(temp_data_path)[b'labels']))
    
    train_data = np.concatenate(temp_data, axis=0)
    train_labels = np.concatenate(temp_labels, axis=0)
    return train_data, train_labels

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # Specify root directory of project
    base_dir = input_filepath
    # Specify path to which data that will be downloaded
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    # Download raw data
    logger.info(f'downloading CIFAR-10 data to {raw_data_dir}')
    download_data(raw_data_dir, url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    logger.info('downloaded data')
    # Extract raw data
    logger.info(f'starting extraction of data to {raw_data_dir}')
    extract_data(raw_data_dir)
    logger.info('extracted data')

    # Set CIFAR-10 data path 
    cifar_10_data_path = os.path.join(raw_data_dir, "cifar-10-batches-py")
    # Load train data
    train_data, train_labels = load_train_data(cifar_10_data_path) 
    # Load meta data
    meta_data_dict = unpickle(os.path.join(cifar_10_data_path, 'batches.meta'))
    # Load test data
    test_data_dict = unpickle(os.path.join(cifar_10_data_path, 'test_batch')) 
    print(train_data.shape, train_labels.shape)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
