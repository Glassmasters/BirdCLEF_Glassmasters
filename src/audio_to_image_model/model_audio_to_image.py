import torch
import torch.nn as nn
import pandas as pd

from audio_to_image_model_def import CustomCNN
from audio_to_image_dataset_def import BirdDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")


def load_metadata(file_path):
    df = pd.read_csv(file_path)
    num_classes = df['primary_label'].nunique()
    return df, num_classes


DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"

# Load the metadata file
metadata_df, num_classes = load_metadata(r"train_metadata_subset.csv")

bird_dataset = BirdDataset(metadata_df, DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR, fixed_length=300, num_classes=num_classes)

train_ratio = 0.8
train_size = int(train_ratio * len(bird_dataset))
val_size = len(bird_dataset) - train_size

train_dataset, val_dataset = random_split(bird_dataset, [train_size, val_size])

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))
model = CustomCNN(num_classes).to(device)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# model.apply(init_weights) #TODO: Check if this is needed

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate on the validation set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

# TODO: Fix the bug in the dataset class/Dimenson mismatch??

# TODO: Different optimizer and loss function
# TODO: Different augmentation techniques
# TODO: Other Model?
# TODO: DIfferent method to handle inblaslance audio length
