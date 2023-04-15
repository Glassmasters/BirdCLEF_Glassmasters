import pandas as pd
from sklearn.utils import resample
from src.plotting.plot_class_occurences import plot_class_occurrences


def balance_dataset(metadata_df, sampling_strategy='auto', target_samples=None):
    """
    Perform resampling on the dataset to handle class imbalance.

    :param metadata_df: A DataFrame containing the columns 'filename' and 'primary_label'
    :param sampling_strategy: The sampling strategy to be used. Either 'auto', 'oversample', 'undersample', or a float.
                              If a float, it should be in the range (0, 1) and represents the desired ratio of the
                              minority class to the majority class after resampling.
                              Default is 'auto', which uses oversampling for the minority class.
    :param target_samples: The target number of samples for each class after resampling. If set, it will override
                           the sampling_strategy parameter.
    :return: A balanced DataFrame
    """
    if target_samples is None:
        if sampling_strategy == 'auto':
            sampling_strategy = 'oversample'

        grouped = metadata_df.groupby('primary_label')
        class_counts = grouped.size()

        if sampling_strategy == 'oversample':
            target_samples = class_counts.max()
        elif sampling_strategy == 'undersample':
            target_samples = class_counts.min()
        else:
            target_samples = int(class_counts.max() * sampling_strategy)

    resampled_dfs = []
    for label, group in metadata_df.groupby('primary_label'):
        resampled_group = resample(group,
                                   replace=len(group) < target_samples,
                                   n_samples=target_samples,
                                   random_state=42)
        resampled_dfs.append(resampled_group)

    balanced_metadata_df = pd.concat(resampled_dfs, ignore_index=True)
    return balanced_metadata_df


if __name__ == '__main__':
    # Base path of the dataset
    DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
    # Load the metadata file
    metadata_df = pd.read_csv(DATASET_BASE_FILE_PATH + r"\train_metadata.csv")

    # Balance the dataset with a target number of samples per class
    balanced_metadata_df = balance_dataset(metadata_df, target_samples=100)

    # Save the balanced metadata file
    balanced_metadata_df.to_csv(r"../../../data/train_metadata_subset_balanced.csv", index=False)

    # Plot the occurrences of each class in the balanced dataset
    plot_class_occurrences(balanced_metadata_df)
