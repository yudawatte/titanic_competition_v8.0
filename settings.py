"""
Contain data paths and namings.
"""
import os

class Settings():
    """A class to store all settings for the titanic competition"""

    def __init__(self):
        self.INPUT_DATA_PATH = os.path.join("data", "inputs")
        self.PROCESSED_DATA_PATH = os.path.join("data", "processed_data")
        self.ORDINAL_ENCODED_DATA_PATH = os.path.join("data", "ordinal_encoded_data")
        self.ONEHOT_ENCODED_DATA_PATH = os.path.join("data", "one_hot_encoded_data")
        #self.SAVE_MODEL_PATH = os.path.join("data", "models")
        #self.RESULT_DATA_PATH = os.path.join("data", "results")
        #self.SAVE_MODEL_PATH = os.path.join("data", "models_v7.0")
        #self.RESULT_DATA_PATH = os.path.join("data", "results_v7.0")
        self.SAVE_MODEL_PATH = os.path.join("data", "models_v8.0")
        self.RESULT_DATA_PATH = os.path.join("data", "results_v8.0")
        self.TRAIN_SET_FILENAME = "train.csv"
        self.TRAIN_TARGET_SET_FILENAME = "target.csv"
        self.TEST_SET_FILENAME = "test.csv"

