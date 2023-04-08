import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import librosa
from preprocess import preprocess_audio
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_metadata(file_path):
    df = pd.read_csv(file_path)
    num_classes = df['primary_label'].nunique()
    return df, num_classes


class CustomCNN(nn.Module):
    def __init__(self, num_classes=264):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 6))
        )

        self.fc1 = nn.Linear(1024 * 6, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class BirdDataset(Dataset):
    def __init__(self, metadata_df, transform=None, fixed_length=300):
        self.metadata_df = metadata_df
        self.transform = transform
        self.fixed_length = fixed_length
        self.base_file_path = DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR
        self.label_to_index = {label: index for index, label in enumerate(sorted(self.metadata_df['primary_label'].unique()))}

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        file_path = row['filename']
        primary_label = row['primary_label']

        # Load and preprocess the audio
        full_file_path = self.base_file_path + "\\" + file_path
        mel_spectrogram = preprocess_audio(full_file_path)
        #mel_spectrogram = torch.unsqueeze(torch.tensor(mel_spectrogram), 0)
        # Pad or truncate the mel spectrogram to a fixed length
        mel_spectrogram = self.pad_or_truncate(mel_spectrogram, self.fixed_length)

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        one_hot_label = self.label_to_onehot(primary_label, num_classes)
        return mel_spectrogram, one_hot_label

    def pad_or_truncate(self, mel_spectrogram, length):
        mel_spectrogram = torch.squeeze(mel_spectrogram)
        if mel_spectrogram.shape[1] < length:
            padding = length - mel_spectrogram.shape[1]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding), 'constant', 0)
        else:
            mel_spectrogram = mel_spectrogram[:, :length]

        mel_spectrogram = torch.unsqueeze(mel_spectrogram, 0)
        return mel_spectrogram

    def label_to_onehot(self, label, num_classes):
        one_hot = torch.zeros(num_classes)
        index = self.label_to_index[label]
        one_hot[index] = 1
        return one_hot


# Real programming starts here

DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"

# Load the metadata file
metadata_df, num_classes = load_metadata(r"train_metadata_subset.csv")

bird_dataset = BirdDataset(metadata_df)

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

#model.apply(init_weights) #TODO: Check if this is needed

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

