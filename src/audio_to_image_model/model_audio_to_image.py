import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import warnings
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from audio_to_image_model_def import CustomCNN, PretrainedBirdClassifier, ImprovedCustomCNN, PretrainedEfficientNetBirdClassifier
from audio_to_image_dataset_def import BirdDataset
from src.preprocess.audio_to_image.data_augmentation import AugmentMelSpectrogram
from src.plotting.plot_training_curve import plot_loss_and_accuracy

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
    correct = 0
    predictions = 0

    progress_bar = tqdm(dataloader)

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Get the predicted class labels
        pred = output.argmax(dim=1, keepdim=True)
        # Convert the one-hot encoded target back to class labels
        target_labels = target.argmax(dim=1, keepdim=True)

        correct += pred.eq(target_labels.view_as(pred)).sum().item()
        predictions += data.shape[0]

        loss = loss.item()
        accuracy = 100 * correct / predictions

        progress_bar.set_description(f"Epoch [{epoch}/{total_epochs}]")
        progress_bar.set_postfix(loss=loss, accuracy=accuracy)


    return loss, accuracy


def _test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # model.eval()
    total_loss = 0

    progress_bar = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target.squeeze(1))
            total_loss += loss.item()

            test_loss /= num_batches
            # Get the predicted class labels
            pred = output.argmax(dim=1, keepdim=True)
            # Convert the one-hot encoded target back to class labels
            target_labels = target.argmax(dim=1, keepdim=True)

            correct += pred.eq(target_labels.view_as(pred)).sum().item()

        print(f"\nTest Loss:{total_loss / size:>8f}  Accuracy: {(100 * correct / size):>0.1f}%\n")

    return total_loss / size, 100 * correct / size


def number_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")


if __name__ == '__main__':

    # Define the dataset base path and the train set file directory
    DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
    TRAIN_SET_FILE_DIR = r"\train_audio"

    # Load the metadata file
    # metadata_df, num_classes = load_metadata(r"../../data/local_subset.csv")
    metadata_df, num_classes = load_metadata(r"../../data/local_subset_stratified.csv")

    # Create a custom dataset object
    augmentations = AugmentMelSpectrogram()
    bird_dataset = BirdDataset(metadata_df, DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR, num_classes=num_classes, transform=augmentations, efficientnet=True)

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
    model = PretrainedEfficientNetBirdClassifier(num_classes).to(device)
    print(model)
    number_trainable_params(model)

    # Initialize the model weights
    model.apply(init_weights)

    # Define the loss function and the optimizer
    # criterion = nn.BCELoss()
    # FIXME: New loss does not need Sigmoid/softmax?
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        # Train
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, epoch + 1, num_epochs)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on the validation set
        val_loss, val_accuracy = _test(val_loader, model, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # Save the model
    torch.save(model.state_dict(), "../../models/model2.pth")
    print("Saved PyTorch Model State to model.pth")

    # Plot the training curve
    plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies)

    # TODO: Different optimizer and loss function
    # TODO: Different augmentation techniques
    # TODO: Other Model? Pretrained model?
    # TODO: DIfferent method to handle inblaslance audio length

# -> kein threshold
# --> Stack 5 second blocks
# --> early stopping
# --> accuracy reset
# --> Audio feature extraction
