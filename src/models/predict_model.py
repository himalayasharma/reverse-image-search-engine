import os
import logging
import pickle
from pathlib import Path
from tensorflow import keras
from generate_encodings import build_intermediate_layer_model
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_query_image(query_img_path):

    image = Image.open(query_img_path)
    image = image.resize((32, 32),Image.Resampling.LANCZOS)
    image_arr = np.array(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr = preprocess_input(image_arr)
    return image, image_arr

def get_idx_of_matches(all_encodings, query_encoding, n_matches=5):
    
    encoding_diff = np.sum(np.abs(all_encodings - query_encoding), axis=1)
    encoding_diff_df = pd.DataFrame(encoding_diff, columns=['absolute_difference'])
    match_idx = encoding_diff_df.sort_values(by=['absolute_difference']).index[1:n_matches+1]
    return match_idx

def main(base_dir):

    logger = logging.getLogger(__name__)

    # -------------- Get query image path -----------------
    print("NOTE: Put query image in 'query-images' directory of project")
    query_img_name = input("Enter query image name (with extension):")
    query_img_path = os.path.join(base_dir, f'query-images/{query_img_name}')
    if(os.path.exists(query_img_path) == False):
        print(f"This image does not exist!")
        return -1
    logger.info('query image successfully loaded')

    # -------------- Pre-process query image -----------------
    query_image_rescaled, query_img_arr = preprocess_query_image(query_img_path)

    # -------------- Load all encodings -----------------
    encodings_path = os.path.join(base_dir, 'models/encodings')
    with open(encodings_path, 'rb') as file_pi:
        all_encodings = pickle.load(file_pi)
    logger.info(f'loaded all image encodings')
    
    # -------------- Load model -----------------
    model_path = os.path.join(base_dir, 'models')
    model = keras.models.load_model(model_path) 
    logger.info(f'loaded model from {model_path}')
    intermediate_layer_model = build_intermediate_layer_model(logger, model)

    # -------------- Create encoding for query image -----------------
    encoding_query_img = intermediate_layer_model.predict(query_img_arr)

    # -------------- Get indices of matching images -----------------
    match_idx = get_idx_of_matches(all_encodings, encoding_query_img, n_matches=5)

    # -------------- Load data -----------------
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    data_dict_path = os.path.join(processed_data_dir, 'data_dict')
    with open(data_dict_path, 'rb') as file_pi:
        data_dict = pickle.load(file_pi)
    all_X = np.concatenate([data_dict['X_train'], data_dict['X_valid'], data_dict['X_test']], axis=0)  

    # -------------- Show query image and top 5 matches -----------------
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(15)
    ax = dict()
    loc_list = [(0,2), (1,0), (1,1), (1,2), (1,3), (1,4)]
    for i, loc in enumerate(loc_list):
        if(i == 0):
            ax[i] = plt.subplot2grid(shape=(2,5), loc=loc, colspan=1)
            ax[i].imshow(query_image_rescaled)
            ax[i].axis('off')
            ax[i].set_title('Rescaled Query Image')
            idx = 0
        else:
            ax[i] = plt.subplot2grid(shape=(2,5), loc=loc, colspan=1)
            ax[i].axis("off")
            ax[i].set_title(f'Match {idx+1}')
            ax[i].imshow(all_X[match_idx[idx]])
            idx += 1
    plt.show()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(project_dir)