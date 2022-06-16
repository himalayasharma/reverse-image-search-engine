import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_sample_images(X, y, y_str, img_save_path):

    idx_for_plotting = np.array([])
    for label in set(y):
        class_img_idx = np.where(y == label)[0]
        some_class_img_idx = np.random.choice(class_img_idx, size=8)
        idx_for_plotting = np.concatenate((idx_for_plotting, some_class_img_idx), axis=None)

    ncols = 8
    nrows = 10
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(ncols*3, nrows*3)
    idx_for_plotting = idx_for_plotting.astype(int)

    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            ax[row, col].axis("off")
            label = y_str[idx_for_plotting[idx]]
            ax[row, col].set_title(f"Label:{label}", fontsize=15)
            ax[row, col].imshow(X[idx_for_plotting[idx]])
            idx += 1
    plt.savefig(os.path.join(img_save_path, 'cifar-10-samples.png'), bbox_inches='tight')

def main(base_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # -------------- Load X, y and corresponding string encoded labels -----------------
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    with open(os.path.join(processed_data_dir, 'all_y'), 'rb') as file_pi:
        all_y = pickle.load(file_pi)
    with open(os.path.join(processed_data_dir, 'all_X'), 'rb') as file_pi:
        all_X = pickle.load(file_pi)
    with open(os.path.join(processed_data_dir, 'all_str_labels'), 'rb') as file_pi:
        all_y_str = pickle.load(file_pi)
    logger.info('loaded data and corresponding strings labels')

    # -------------- Save CIFAR-10 sample images -----------------
    img_save_path = os.path.join(base_dir, 'reports/figures')
    save_sample_images(all_X, all_y, all_y_str, img_save_path)
    logger.info(f'saved sample CIFAR-10 images (with labels) at {img_save_path}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)



