# This file defines how to handle the imbalance of the dataset

import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt


def balance_dataset(metadata_df, sampling_strategy='auto'):
    """
    Perform resampling on the dataset to handle class imbalance.

    :param metadata_df: A DataFrame containing the columns 'filename' and 'primary_label'
    :param sampling_strategy: The sampling strategy to be used. Either 'auto', 'oversample', 'undersample', or a float.
                              If a float, it should be in the range (0, 1) and represents the desired ratio of the
                              minority class to the majority class after resampling.
                              Default is 'auto', which uses oversampling for the minority class.
    :return: A balanced DataFrame
    """
    if sampling_strategy == 'auto':
        sampling_strategy = 'oversample'

    grouped = metadata_df.groupby('primary_label')
    class_counts = grouped.size()

    if sampling_strategy == 'oversample':
        max_class_count = class_counts.max()
    elif sampling_strategy == 'undersample':
        max_class_count = class_counts.min()
    else:
        max_class_count = int(class_counts.max() * sampling_strategy)

    resampled_dfs = []
    for label, group in grouped:
        resampled_group = resample(group,
                                   replace=True,
                                   n_samples=max_class_count,
                                   random_state=42)
        resampled_dfs.append(resampled_group)

    balanced_metadata_df = pd.concat(resampled_dfs, ignore_index=True)
    return balanced_metadata_df


def plot_class_occurrences(metadata_df):
    class_counts = metadata_df['primary_label'].value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_xlabel('Class')
    ax.set_ylabel('Occurrences')
    ax.set_title('Occurrences of Classes in the Balanced Dataset')
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':
    # Base path of the dataset
    DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
    # Load the metadata file
    metadata_df = pd.read_csv(DATASET_BASE_FILE_PATH + r"\train_metadata.csv")

    # Balance the dataset
    balanced_metadata_df = balance_dataset(metadata_df)

    # Save the balanced metadata file
    balanced_metadata_df.to_csv(r"train_metadata_subset_balanced.csv", index=False)

    # Plot the occurrences of each class in the balanced dataset
    plot_class_occurrences(balanced_metadata_df)
