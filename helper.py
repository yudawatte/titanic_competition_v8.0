"""
Contain common functionalities which may commonly used in all modules
"""
import os
from settings import Settings
import pickle
import pandas as pd

def save_model(model, model_name):
    """
    Save model
    :param model: trained model need to be saved
    :param model_name: trained model name
    :return:
    """
    print("\tSave model: ", model_name)
    sett = Settings()
    try:
        os.mkdir(sett.SAVE_MODEL_PATH)
        print("\t", sett.SAVE_MODEL_PATH)
    except OSError as error:
        print("\t", sett.SAVE_MODEL_PATH)

    file_name = os.path.join(sett.SAVE_MODEL_PATH, model_name)
    pickle.dump(model, open(file_name, 'wb'))

def load_model(model_name):
    """
    Load model
    :param model_name: model name need to be loaded
    :return: loaded model
    """
    print("\tLoad model: ", model_name)
    sett = Settings()
    file_name = os.path.join(sett.SAVE_MODEL_PATH, model_name)
    loaded_model = pickle.load(open(file_name, 'rb'))

    return loaded_model

def show_results(cv):
    """
    Display cross validation results
    :param cv: cross validation result
    :return:
    """
    print("Scross validation scores: ", cv)
    print("Scross validation mean: ", cv.mean())

def read_data(path, file_name):
    """
    Read input csv files
    :param path: csv file path
    :param file_name: csv file name
    :return: csv content as a dataframe
    """
    dataset = pd.read_csv(os.path.join(path, file_name))
    return dataset

def save_data(data_set, path, filename, index=False, header=False):
    """
    Write to a csv
    :param data_set: data set need to be saved
    :param path: file saving location
    :param filename: file name
    :param index: required to use instance index
    :param header: require to use data header
    :return:
    """
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, filename)
    pd.DataFrame(data_set).to_csv(path_or_buf=csv_path, index=index, header=header)