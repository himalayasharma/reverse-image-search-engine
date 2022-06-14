import os
import pickle
import logging
import numpy as np
from pathlib import Path
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Build intermediate layer model
def build_intermediate_layer_model(logger, model):

    from keras.models import Model
    intermediate_layer_model = Model(inputs=model.input, 
                                    outputs=model.get_layer('dense_2').output)
    logger.info("constructed intermediate layer model")
    return intermediate_layer_model

# Generate encodings for all images in dataset
def generate_encodings(logger, model, data_dict):

    logger.info(f"started generating encodings")
    all_X = np.concatenate([data_dict['X_train'], data_dict['X_valid'], data_dict['X_test']], axis=0)
    all_encodings = model.predict(all_X)
    logger.info(f"generated encodings for all {len(all_X)} images in CIFAR-10 dataset")
    return all_encodings

def main(base_dir):

    logger = logging.getLogger(__name__)

    # Load data
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    data_dict = dict()
    data_str_list = ['X_train', 'y_train', 'X_valid', 'y_valid', 'X_test', 'y_test']
    for data_str in (data_str_list):
        data_dict[data_str] = np.load(os.path.join(processed_data_dir, f"{data_str}.npy"))

    # Load trained model
    model_path = os.path.join(base_dir, 'models')
    model = keras.models.load_model(model_path) 
    logger.info(f'loaded model from {model_path}')

    # Build model with intermediate layer
    intermediate_layer_model = build_intermediate_layer_model(logger, model)

    # Generate encodings
    all_encodings = generate_encodings(logger, intermediate_layer_model, data_dict)
    # Save encodings
    prediction_path = os.path.join(model_path, 'predictions')
    with open(prediction_path, 'wb') as file_pi:
        pickle.dump(all_encodings, file_pi)
    logger.info(f'saved encodings to {prediction_path}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(project_dir)