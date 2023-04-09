import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from clearml import Task, Dataset

# Initialize ClearML Task
task = Task.init(project_name="BirdCLEF_Glassmasters", task_name="create_train_data_subset")

DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"

# Load the metadata file
train_metadata = pd.read_csv(DATASET_BASE_FILE_PATH + r"\train_metadata.csv")

# Split the data into a subset and the full training set using GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
train_idx, full_idx = next(gss.split(train_metadata, groups=train_metadata["primary_label"]))

train_metadata_subset = train_metadata.iloc[train_idx]
train_metadata_full = train_metadata.iloc[full_idx]

# Save the subset and full training set metadata to CSV files
train_metadata_subset.to_csv("train_metadata_subset.csv", index=False)
train_metadata_full.to_csv("train_metadata_full.csv", index=False)

# Create a ClearML Dataset for the subset and full training set metadata
subset = Dataset.create(dataset_name='subset', dataset_project='BirdCLEF_Glassmasters')
subset.add_files('train_metadata_subset.csv')
subset.upload()
subset.finalize()

full = Dataset.create(dataset_name='full', dataset_project='BirdCLEF_Glassmasters')
full.add_files('train_metadata_full.csv')
full.upload()
full.finalize()

# Log the results
task.get_logger().report_table(title="Subset Metadata", series="Metadata", iteration=0, table_plot=train_metadata_subset.head())
task.get_logger().report_table(title="Full Metadata", series="Metadata", iteration=0, table_plot=train_metadata_full.head())
