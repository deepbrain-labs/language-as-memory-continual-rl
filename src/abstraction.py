import torch
import torch.nn as nn
import numpy as np

class AbstractionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_abstraction(model, X, Y, epochs=10, lr=1e-3):
    import torch.optim as optim
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    Y_t = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, Y_t)
        loss.backward()
        optimizer.step()
    return model
