import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import warnings
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from audio_model_def import Custom1dCNN, Custom1dCNNSmall, Custom1dCNNOnPretrainedModel
from audio_dataset_def import BirdDatasetAudioOnly

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


def number_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")


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
            # Get the predicted class labels
            pred = output.argmax(dim=1, keepdim=True)
            # Convert the one-hot encoded target back to class labels
            target_labels = target.argmax(dim=1, keepdim=True)

            correct += pred.eq(target_labels.view_as(pred)).sum().item()

        print(f"\nTest Loss:{total_loss / size:>8f}  Accuracy: {(100 * correct / size):>0.1f}%\n")


if __name__ == '__main__':
    # Define the dataset base path and the train set file directory
    DATASET_BASE_FILE_PATH = r"D:\Datasets\birdclef-2023"
    TRAIN_SET_FILE_DIR = r"\train_audio_balanced"

    use_pretrained_model = False
    use_small_1d_cnn = True
    if use_pretrained_model:
        model_name = "audio_1d_CNN_model_on_feature_extractor.pth"
        # MODEL_WEIGHTS = f"../../models/{model_name}"
    elif use_small_1d_cnn:
        model_name = "audio_1d_CNN_model_small.pth"
    else:
        model_name = "audio_1d_CNN_model.pth"
        MODEL_WEIGHTS = f"../../models/{model_name}"

    # Load the metadata file
    metadata_df, num_classes = load_metadata(DATASET_BASE_FILE_PATH + r"\output_classes_distribution.csv")

    # Create a custom dataset object
    bird_dataset = BirdDatasetAudioOnly(metadata_df, DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR, 
                                        num_classes=num_classes, 
                                        use_pretrained_feature_extractor=use_pretrained_model)

    # Split the dataset into train and validation sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(bird_dataset))
    val_size = len(bird_dataset) - train_size

    train_dataset, val_dataset = random_split(bird_dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize GPU device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    # Initialize the model
    if use_pretrained_model:
        model = Custom1dCNNOnPretrainedModel(num_classes).to(device)
    elif use_small_1d_cnn:
        model = Custom1dCNNSmall(num_classes).to(device)
    else:
        model = Custom1dCNN(num_classes).to(device)
        if MODEL_WEIGHTS:
            model.load_state_dict(torch.load(MODEL_WEIGHTS))
    number_trainable_params(model)

    # Initialize the model weights
    # model.apply(init_weights)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        # Train
        train(train_loader, model, criterion, optimizer, epoch +1, num_epochs)
        # Evaluate on the validation set
        _test(val_loader, model, criterion)

    torch.save(model.state_dict(), f"../../models/{model_name}")
    print(f"Saved PyTorch Model State to {model_name}")
    