import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

# Define the device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






class CNN_LSTM_Model(nn.Module):
    def __init__(self,numStock,numFeature):
        super(CNN_LSTM_Model, self).__init__() 
        self.numStock = numStock       
        self.numFeature = numFeature    
        print(f"self.numStock{self.numStock},self.numFeature{self.numFeature}")   

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.numFeature, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=1, hidden_size=26, num_layers=1, batch_first=True)

        # Replace max pooling with adaptive avg pooling
        self.pool = nn.MaxPool1d(1)  
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(26, 1)
        

    def forward(self, x):
        # x = x.unsqueeze(1)  # Add a channel dimension
        x = x.reshape(-1,self.numFeature,self.numStock)
        x = self.cnn(x) 
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.view(-1)
        return x






class NNPredictionModel:
    def __init__(self, numFeature, numStock):
        self.numFeature = numFeature
        self.numStock = numStock
        self.model = CNN_LSTM_Model(numStock,numFeature)
        # self.model = CNN_LSTM_Model().to(device) 


    def train(self, X_train, y_train, epochs=500, lr=0.0001):
    # def train(self, X_train, y_train, epochs=2000, lr=0.0001):
    # def train(self, X_train, y_train, epochs=2000, lr=0.001):
    # def train(self, X_train, y_train, epochs=20000, lr=0.01):
    # def train(self, X_train, y_train, epochs=200, lr=0.001):
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        y_train = y_train.squeeze(-1)  # Ensuring the target shape matches the output shape

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # def test(self, X_test, y_test):
    #     with torch.no_grad():
    #         y_test = y_test.squeeze(-1)  # Ensuring the target shape matches the output shape
    #         outputs = self.model(X_test)
    #         loss = F.mse_loss(outputs, y_test)
    #         return loss.item()

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(X)
            return y_pred

if __name__=="__main__":
    
    # numStock = 1
    numStock = 100
    # numStock = 492
    numFeature = 40

    # Sample data
    X_train = torch.randn(260*numStock, numFeature)
    y_train = torch.randn(260*numStock, 1)
    X_test = torch.randn(26*numStock, numFeature)  # Test data
    y_test = torch.randn(26*numStock, 1)
    


    stock_prediction_model = NNPredictionModel(numFeature, numStock)
    stock_prediction_model.train(X_train, y_train)
    test_loss = stock_prediction_model.test(X_test, y_test)
    y_pred = stock_prediction_model.predict(X_test)  # y_pred as the output
    print(f'Test Loss (MSE): {test_loss}')
    
    # summary(stock_prediction_model.model,X_train.shape)
