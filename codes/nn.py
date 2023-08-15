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
y_train = torch.randn(260*numStock, 1)
X_test = torch.randn(26*numStock, numFeature)  # Test data
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


class NNPredictionModel:
    def __init__(self, numFeature, numStock):
        self.numFeature = numFeature
        self.numStock = numStock
        self.model = CNN_LSTM_Model()

    def train(self, X_train, y_train, epochs=50, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def test(self, X_test, y_test):
        with torch.no_grad():
            outputs = self.model(X_test)
            loss = F.mse_loss(outputs, y_test)
            return loss.item()

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(X)
            return y_pred

if __name__=="__main__":
    stock_prediction_model = NNPredictionModel(numFeature, numStock)
    stock_prediction_model.train(X_train, y_train)
    test_loss = stock_prediction_model.test(X_test, y_test)
    y_pred = stock_prediction_model.predict(X_test)  # y_pred as the output
    print(f'Test Loss (MSE): {test_loss}')
