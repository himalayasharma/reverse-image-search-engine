import os
import pickle
import logging
import numpy as np
from pathlib import Path
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_intermediate_layer_model(logger, model):

    from keras.models import Model
    intermediate_layer_model = Model(inputs=model.input, 
                                    outputs=model.get_layer('dense_2').output)
    logger.info("constructed intermediate layer model")
    return intermediate_layer_model

def generate_encodings(logger, model, data_dict):

    logger.info(f"started generating encodings")
    all_X = np.concatenate([data_dict['X_train'], data_dict['X_valid'], data_dict['X_test']], axis=0)
    all_encodings = model.predict(all_X)
    logger.info(f"generated encodings for all {len(all_X)} images in CIFAR-10 dataset")
    return all_encodings

def unpickle(file):

    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def get_str_labels(logger, data_dict):

    meta_data_dict = unpickle(os.path.join('data/raw/cifar-10-batches-py', 'batches.meta'))
    mapping_labels = dict(zip(np.unique(data_dict['y_train']), meta_data_dict[b'label_names']))
    convert_labels = lambda label:mapping_labels[label].decode('ascii')
    all_y = np.concatenate([data_dict['y_train'], data_dict['y_valid'], data_dict['y_test']], axis=0)
    all_y_str = np.array(list(map(convert_labels, all_y)))
    return all_y_str
    
def main(base_dir):

    logger = logging.getLogger(__name__)

    # -------------- Load data -----------------
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    data_dict_path = os.path.join(processed_data_dir, 'data_dict')
    with open(data_dict_path, 'rb') as file_pi:
        data_dict = pickle.load(file_pi)
    logger.info('loaded train, valid and test data')

    # -------------- Load trained model -----------------
    model_path = os.path.join(base_dir, 'models')
    model = keras.models.load_model(model_path) 
    logger.info(f'loaded model from {model_path}')

    # -------------- Build intermediate layer model -----------------
    intermediate_layer_model = build_intermediate_layer_model(logger, model)

    # -------------- Generate encodings for all images -----------------
    all_encodings = generate_encodings(logger, intermediate_layer_model, data_dict)

    # -------------- Save encodings for all images -----------------
    encodings_path = os.path.join(model_path, 'encodings')
    with open(encodings_path, 'wb') as file_pi:
        pickle.dump(all_encodings, file_pi)
    logger.info(f'saved encodings to {encodings_path}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(project_dir)