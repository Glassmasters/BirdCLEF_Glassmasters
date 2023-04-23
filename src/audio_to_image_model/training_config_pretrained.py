# This file contains the configuration for training a pretrained model

import pandas as pd


def load_metadata(file_path):
    df = pd.read_csv(file_path)
    num_classes = df['primary_label'].nunique()
    return df, num_classes


metadata_df, num_classes = load_metadata(r"../../data/local_subset_stratified.csv")
# Define the dataset base path and the train set file directory
DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"


dataset_config = {"metadata_df": metadata_df,
                  "train_set_file_dir": DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR,
                  "fixed_length": 5,
                  "num_classes": num_classes,
                  "transform": None,
                  "efficientnet": True
                  }