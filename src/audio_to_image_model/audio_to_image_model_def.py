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
