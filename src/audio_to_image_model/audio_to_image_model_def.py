import torch.nn as nn
from torchvision import models
import torch


# Model Definition
class CustomCNN(nn.Module):
    """
    A custom convolutional neural network (CNN) model for bird sound classification.
    """

    def __init__(self, num_classes=264):
        """
        Initialize the model
        Args:
            num_classes (int, optional): The number of bird species. Defaults to 264.
        """
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
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
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


class PretrainedBirdClassifier(nn.Module):
    """
    A bird sound classification model using a pre-trained CNN.
    """

    def __init__(self, num_classes=264, trainable=False):
        """
        Initialize the model.
        Args:
            num_classes (int, optional): The number of bird species. Defaults to 264.
            trainable (bool, optional): If True, the pre-trained model's weights will be updated during training.
        """
        super(PretrainedBirdClassifier, self).__init__()

        # Load a pre-trained ResNet model and remove the final classification layer
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # Set the requires_grad attribute based on the trainable parameter
        for param in self.base_model.parameters():
            param.requires_grad = trainable

        # Add a custom classification layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class ImprovedCustomCNN(nn.Module):
    """
    An improved custom convolutional neural network (CNN) model for bird sound classification,
    based on the input melspectrogram.
    """

    def __init__(self, num_classes=264):
        """
        Initialize the model
        Args:
            num_classes (int, optional): The number of bird species. Defaults to 264.
        """
        super(ImprovedCustomCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor (melspectrogram)
        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x


from torchvision.models import resnet18

class PretrainedBirdClassifier(nn.Module):
    """
    A bird sound classification model based on a pre-trained ResNet-18.
    """

    def __init__(self, num_classes=264):
        """
        Initialize the model
        Args:
            num_classes (int, optional): The number of bird species. Defaults to 264.
        """
        super(PretrainedBirdClassifier, self).__init__()

        # Load pre-trained ResNet-18 model
        self.base_model = resnet18(pretrained=True)

        # Replace the first convolutional layer to accept single-channel input (melspectrogram)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last fully connected layer for bird species classification
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor (melspectrogram)
        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x = self.base_model(x)
        return x
