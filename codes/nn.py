import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# numStock = 1
numStock = 100
# numStock = 492
numFeature = 40


# Sample data
X_train = torch.randn(260*numStock, numFeature)
y_train = torch.randn(26*numStock, 1)
X_test = torch.randn(260*numStock, numFeature)  # Test data
y_test = torch.randn(26*numStock, 1)


class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(numFeature, 64, kernel_size=3, stride=1, padding=1), # Use numFeature here
            nn.ReLU(),
        )

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=50, num_layers=1, batch_first=True)

        # Adding a max pooling layer to reduce the sequence length by a factor of 10
        self.pool = nn.MaxPool1d(10)  # Pooling with a kernel size of 10

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.view(260 * numStock, numFeature).unsqueeze(0) # Use numFeature here
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x.squeeze(0)



def train():
    # Model, loss function, and optimizer
    model = CNN_LSTM_Model()  # No need to pass numStock
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
    return model
def test(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        loss = F.mse_loss(outputs, y_test)
        return loss.item()


if __name__=="__main__":
    trained_model = train()
    test_loss = test(trained_model, X_test, y_test)
    print(f'Test Loss (MSE): {test_loss}')


refactor the class of nn

and build another bigger class to
take X_train, y_train, X_test as input and y_pred as output
