import os
import logging
import pickle
from pathlib import Path
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_model(logger, data_dict):

    from tensorflow.keras.applications.vgg16 import VGG16
    # Load pre-trained model for transfer learning
    base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=data_dict["X_train"][0].shape)
    logger.info('loaded pre-trained frozen VGG-16 model with imagenet weights')

    # Add dense layers at the end of pre-trained model
    base_model.trainable = True
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense

    model = Sequential([
        base_model,
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=10)
    ])

    logger.info('added 4 dense layers to make the network compatible for current use-case')
    print(model.summary())

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics='accuracy')

    return model

def main(base_dir):

    logger = logging.getLogger(__name__)

    # -------------- Load processed data -----------------
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    data_dict_path = os.path.join(processed_data_dir, 'data_dict')
    with open(data_dict_path, 'rb') as file_pi:
        data_dict = pickle.load(file_pi)
    logger.info('loaded train, valid and test data')
    
    # -------------- Pre-process data -----------------
    from tensorflow.keras.applications.vgg16 import preprocess_input

    data_dict['X_train'] = preprocess_input(data_dict['X_train'])
    data_dict['X_valid'] = preprocess_input(data_dict['X_valid'])
    data_dict['X_test'] = preprocess_input(data_dict['X_test'])

    # -------------- Build and compile model -----------------
    model = build_model(logger, data_dict)

    # -------------- Train model -----------------
    epochs = int(input("Enter no. of epochs:"))
    logger.info(f'training model for {epochs} epochs')
    history = model.fit(data_dict['X_train'], data_dict['y_train'],\
        epochs=epochs, validation_data=(data_dict['X_valid'], data_dict['y_valid']), batch_size=32)

    # -------------- Save history -----------------
    model_path =  os.path.join(base_dir, 'models')
    if(os.path.exists(model_path) == False):
        os.makedirs(model_path)
    history_path = os.path.join(model_path, 'history')

    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    logger.info('saved performance history')

    # -------------- Save model -----------------
    model.save(model_path)
    logger.info(f'saved model history to {model_path}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(project_dir)