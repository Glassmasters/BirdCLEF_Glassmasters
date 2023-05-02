import torch.nn as nn
import torch


class Custom1dCNN(nn.Module):
    """
    Custom 1d CNN model implementation to classify bird sounds.
    """

    def __init__(self, num_classes: int = 264):
        """
        :param num_classes: int, the number of classes to predict.

        Initializes the 1d CNN model.
        """
        super(Custom1dCNN, self).__init__()

        self.conv1d_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.1),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1),
            nn.Conv1d(32, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8),
            nn.Dropout(p=0.1)
        )

        self.fc1 = nn.Linear(8 * 64, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1d_block(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class Custom1dCNNSmall(nn.Module):
    """
    Custom 1d CNN model implementation to classify bird sounds.
    """

    def __init__(self, num_classes: int = 264):
        """
        :param num_classes: int, the number of classes to predict.

        Initializes the 1d CNN model.
        """
        super(Custom1dCNNSmall, self).__init__()

        self.conv1d_block = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(8),
            nn.Dropout(p=0.1),
            nn.Conv1d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.1),
            nn.Conv1d(16, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8),
            nn.Dropout(p=0.1)
        )

        self.fc1 = nn.Linear(8 * 32, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1d_block(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class Custom1dCNNOnPretrainedModel(nn.Module):
    """
    Small custom 1d CNN model for using it on top of the feature extractor data.
    """

    def __init__(self, num_classes: int = 264):
        """
        :param num_classes: int, the number of classes to predict.

        Initializes the 1d CNN model.
        """
        super(Custom1dCNNOnPretrainedModel, self).__init__()

        self.conv1d_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.AdaptiveMaxPool1d(8),
            nn.Dropout(p=0.1)
        )

        self.fc1 = nn.Linear(8 * 16, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1d_block(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x