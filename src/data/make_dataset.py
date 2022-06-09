# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # Import required packages
    import os, wget, tarfile
    logger.info('imported required packages')
    # Specify root directory of project
    base_dir = input_filepath
    # Specify path to which data that will be downloaded
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    # Make path if does not exits
    if(os.path.exists(raw_data_dir) == False):
        os.makedirs(raw_data_dir)
    logger.info(f'downloading CIFAR-10 data to {raw_data_dir}')
    # Download data if it does not exist on disk
    if(os.path.exists(os.path.join(raw_data_dir, "cifar-10-python.tar.gz")) == False):
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        print("\n")
        wget.download(url, raw_data_dir)
        print("\n")
    logger.info('downloaded data')
    # Extract data
    file_handler = tarfile.open(os.path.join(raw_data_dir, "cifar-10-python.tar.gz"))
    logger.info(f'extracted data to {raw_data_dir}')
    file_handler.extractall(raw_data_dir)
    file_handler.close()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
