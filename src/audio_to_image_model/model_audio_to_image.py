import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import warnings
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from audio_to_image_model_def import CustomCNN
from audio_to_image_dataset_def import BirdDataset
from src.preprocess.audio_to_image.data_augmentation import AugmentMelSpectrogram

warnings.filterwarnings("ignore")

torch.manual_seed(42)


def load_metadata(file_path):
    df = pd.read_csv(file_path)
    num_classes = df['primary_label'].nunique()
    return df, num_classes


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(dataloader, model, loss_fn, optimizer, epoch, total_epochs):
    size = len(dataloader.dataset)
    model.train()

    progress_bar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch_idx * len(data)
        progress_bar.set_description(f"Epoch [{epoch}/{total_epochs}]")
        progress_bar.set_postfix(loss= loss, accuracy= (100 * current / size))


def _test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    total_loss = 0

    progress_bar = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()

            test_loss /= num_batches
            correct /= size
            progress_bar.set_description(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Define the dataset base path and the train set file directory
DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"

# Load the metadata file
metadata_df, num_classes = load_metadata(r"../../data/local_subset.csv")

# Create a custom dataset object
augmentations = AugmentMelSpectrogram()
bird_dataset = BirdDataset(metadata_df, DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR, fixed_length=300, num_classes=num_classes, transform=augmentations or None)

# Split the dataset into train and validation sets
train_ratio = 0.8
train_size = int(train_ratio * len(bird_dataset))
val_size = len(bird_dataset) - train_size

train_dataset, val_dataset = random_split(bird_dataset, [train_size, val_size])

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# Initialize the model
model = CustomCNN(num_classes).to(device)
print(model)

# Initialize the model weights
model.apply(init_weights)

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train
    train(train_loader, model, criterion, optimizer, epoch +1, num_epochs)
    # Evaluate on the validation set
    _test(val_loader, model, criterion)

torch.save(model.state_dict(), "../../models/model.pth")
print("Saved PyTorch Model State to model.pth")

# TODO: Different optimizer and loss function
# TODO: Different augmentation techniques
# TODO: Other Model? Pretrained model?
# TODO: DIfferent method to handle inblaslance audio length
