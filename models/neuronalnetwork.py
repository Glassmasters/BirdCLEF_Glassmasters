from torch import nn, optim
from torch.utils.data import DataLoader


class AudioClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioClassificationModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        sequence_length = x.size(1)
        input_size = x.size(2)

        # Reshape to matrix of shape (batch_size * sequence_length, input_size)
        x = x.view(batch_size * sequence_length, input_size)

        # Pass through GRU
        _, hidden = self.gru(x)

        # Take last hidden state of second layer
        hidden = hidden[-1]

        # Pass through fully connected layer
        out = self.fc1(hidden)

        # Reshape to batch size x num_classes
        out = out.view(batch_size, -1)

        return out


def train_model(model, train_dataset, num_epochs, batch_size, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (audio_clips, labels) in enumerate(train_loader):
            # Reshape audio clips to fixed size of 160000 samples
            audio_clips = nn.utils.rnn.pad_sequence(audio_clips, batch_first=True, padding_value=0.0)
            audio_clips = audio_clips.view(-1, 1, 160000)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(audio_clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0