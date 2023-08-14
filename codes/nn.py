import torch
import torch.nn as nn
import torch.optim as optim

# Sample data (replace with your actual data)
X_train = torch.randn(100, 260, 40)  # 100 samples, 260 time steps, 40 features
y_train = torch.randn(100, 26, 1)  # 100 samples, 26 time steps, 1 feature


class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=50, num_layers=1, batch_first=True)

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to match Conv1d input shape
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Permute back to match LSTM input shape
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)  # Permute for pooling
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Permute back for LSTM
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x


def train():
    # Model, loss function, and optimizer
    model = CNN_LSTM_Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # You can now use the trained model for prediction
if __name__=="__main__":
    train()
