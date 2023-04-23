import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATASET_BASE_FILE_PATH = r"src/birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"
# Load the metadata file
train_metadata = pd.read_csv("../../../data/train_metadata_subset_balanced.csv")
# Split the data into train, test, and validation sets using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(train_metadata, train_metadata["primary_label"]))
train_metadata_temp = train_metadata.iloc[train_idx]
test_metadata = train_metadata.iloc[test_idx]
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss.split(train_metadata_temp, train_metadata_temp["primary_label"]))
train_metadata = train_metadata_temp.iloc[train_idx]
val_metadata = train_metadata_temp.iloc[val_idx]
# Create a local subset for testing (25% of the whole dataset)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
_, local_subset_idx = next(sss.split(train_metadata, train_metadata["primary_label"]))
local_metadata_subset = train_metadata.iloc[local_subset_idx]
# Save the train, test, and validation sets to CSV files
train_metadata.to_csv("../../../data/train_metadata.csv", index=False)
test_metadata.to_csv("../../../data/test_metadata.csv", index=False)
val_metadata.to_csv("../../../data/val_metadata.csv", index=False)
local_metadata_subset.to_csv("../../../data/local_subset_stratified.csv", index=False)