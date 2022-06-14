import os
import logging
from re import L
import numpy as np
import pickle
from pathlib import Path

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
    base_model.trainable = False
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense

    model = Sequential([
        base_model,
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    logger.info('added 3 dense layers to make the network compatible for current use-case')
    print(model.summary())

    # Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

    return model

def main(base_dir):

    logger = logging.getLogger(__name__)

    # Load processed data
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    data_dict = dict()
    data_str_list = ['X_train', 'y_train', 'X_valid', 'y_valid', 'X_test', 'y_test']
    for data_str in (data_str_list):
        data_dict[data_str] = np.load(os.path.join(processed_data_dir, f"{data_str}.npy"))
    data_dict['meta'] = os.path.join(base_dir, 'data/raw')
    logger.info('loaded train, valid, test and data')
    
    # Pre-process data
    from tensorflow.keras.applications.vgg16 import preprocess_input

    data_dict['X_train'] = preprocess_input(data_dict['X_train'])
    data_dict['X_valid'] = preprocess_input(data_dict['X_valid'])
    data_dict['X_test'] = preprocess_input(data_dict['X_test'])
    
    # Build and compile model
    print("=================================================================")
    model = build_model(logger, data_dict)
    print("=================================================================")

    # Train model
    epochs = int(input("Enter no. of epochs:"))
    logger.info(f'training model for {epochs} epochs')
    print("=================================================================")
    history = model.fit(data_dict['X_train'], data_dict['y_train'],\
        epochs=epochs, validation_data=(data_dict['X_valid'], data_dict['y_valid']), batch_size=32)
    print("=================================================================")

    # Save history 
    model_path =  os.path.join(base_dir, 'models')
    if(os.path.exists(model_path) == False):
        os.makedirs(model_path)
    history_path = os.path.join(model_path, 'history')

    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    logger.info('saved performance history')

    # Save model
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
