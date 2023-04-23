import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class NeuronalNetwork:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y):
        # Convert inputs to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        # Train the model
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(X_tensor)
        loss = self.loss_function(y_pred, y_tensor)
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        # Convert inputs to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float()
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        score = accuracy_score(y, predictions)
        return score
