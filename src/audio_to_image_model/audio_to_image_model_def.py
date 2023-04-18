import torch.nn as nn
from torchvision import models
import torch
from torchvision.models import resnet18, efficientnet_b0


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


class PretrainedBirdClassifier(nn.Module):
    """
    A bird sound classification model based on a pre-trained ResNet-18.
    """

    def __init__(self, num_classes=264, fine_tune_last_blocks=True):
        """
        Initialize the model
        Args:
            num_classes (int, optional): The number of bird species. Defaults to 264.
        """
        super(PretrainedBirdClassifier, self).__init__()

        # Load pre-trained ResNet-18 model
        self.base_model = resnet18(pretrained=True)

        # Set the requires_grad attribute based on the fine_tune_last_blocks parameter
        for name, param in self.base_model.named_parameters():
            if fine_tune_last_blocks:
                # Fine-tune the last two residual blocks (layer3 and layer4)
                if "layer3" in name or "layer4" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        # Replace the first convolutional layer to accept single-channel input (melspectrogram)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last fully connected layer for bird species classification
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

        # Define the number of features from the pre-trained model
        num_features = self.base_model.fc.in_features

        # Add a custom classification layer
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
            # nn.Sigmoid()
            #nn.Softmax(dim=1)
        )

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


class PretrainedEfficientNetBirdClassifier(nn.Module):
    """
    A bird sound classification model based on a pre-trained EfficientNet-b0.
    """

    def __init__(self, num_classes=264, fine_tune_last_blocks=True):
        """
        Initialize the model
        Args:
            num_classes (int, optional): The number of bird species. Defaults to 264.
            fine_tune_last_blocks (bool, optional): Whether to fine-tune the last blocks of the base model. Defaults to True.
        """
        super(PretrainedEfficientNetBirdClassifier, self).__init__()

        # Load pre-trained EfficientNet-b0 model
        self.base_model = efficientnet_b0(pretrained=True)

        # Set the requires_grad attribute based on the fine_tune_last_blocks parameter
        for name, param in self.base_model.named_parameters():
            if fine_tune_last_blocks:
                # Fine-tune the last two stages (blocks 5 and 6)
                if any(s in name for s in ["features.6.", "features.7."]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        # Replace the first convolutional layer to accept single-channel input (melspectrogram)
        #self.base_model.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Replace the last fully connected layer for bird species classification
        num_features = self.base_model.classifier[1].in_features

        self.base_model.classifier = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

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
