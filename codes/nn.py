import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define hyperparameters
bin_size = 26
train_days = 50 

# Define LSTM block
class LSTMBlock(nn.Module):
    def __init__(self, numStock):
        super(LSTMBlock, self).__init__()
        self.numStock = numStock
        self.lstm = nn.LSTM(52, bin_size, num_layers=1, batch_first=True).double()  # Because we use batch-first input and need double type
        self.fc1 = nn.Linear(bin_size, 1).double()  # Because converting LSTM output to desired shape and need double type
        self.sigmoid = nn.Sigmoid()  # Because activation function
        
    def forward(self, x):
        out, _ = self.lstm(x)  # Because LSTM output
        out = self.sigmoid(out)  # Because applying activation function to LSTM output
        out = self.fc1(out)  # Because fully connected layer to convert LSTM output to desired shape
        out = out.reshape(self.numStock, train_days * bin_size, 1)  # Because reshaping the output
        return out

# Define MLP block
class MLPBlock(nn.Module):
    def __init__(self, numStock):
        super().__init__()
        self.numStock = numStock
        self.fc = nn.Sequential(
            nn.Linear(52, 130).double(),  # Because input layer: 52 input features, 130 output features and need double type
            nn.ReLU(),  # Because activation function
            nn.Linear(130, 52).double(),  # Because hidden layer: 130 input features, 52 output features and need double type
            nn.Sigmoid()  # Because activation function to ensure output is between 0 and 1
        )
    
    def forward(self, x):
        # print(x.shape)
        x = self.fc(x)  # Because forward pass through MLP block
        x = x.view(self.numStock, train_days * bin_size, 52)  # Because reshaping the output
        return x

# Define main model block
class CNNLSTM(nn.Module):
    def __init__(self, numStock):
        super(CNNLSTM, self).__init__()
        self.mlp = MLPBlock(numStock)
        self.lstm_block = LSTMBlock(numStock)
        
    def forward(self, x):
        x = self.mlp(x)  # Because passing through MLP block
        x = self.lstm_block(x)  # Because passing through LSTM block
        return x

# Calculate number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # Because only counting parameters that require gradients

# Define prediction model class
class NNPredictionModel:
    def __init__(self, numStock, learning_rate=0.001, epochs=10, batch_size=32, debug=True):
        self.model = CNNLSTM(numStock).double()  # Because we need the model to be in double type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Because using Adam optimizer
        self.criterion = nn.MSELoss()  # Because using Mean Squared Error loss for regression task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Because using CUDA or CPU device
        self.debug = debug
        
    def train(self, X_train, y_train):
        self.model.to(self.device)
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        
        train_data = TensorDataset(X_train, y_train)  # Because packing training data into TensorDataset
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)  # Because using DataLoader to load training data
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            #     print(batch_idx)
            #     print(X_batch.shape)
            #     print(y_batch.shape)
                self.optimizer.zero_grad()
                
                if self.debug: forward_start = time.time()
                outputs = self.model(X_batch.double())  # Because converting to double type
                if self.debug: forward_time = time.time() - forward_start
                
                if self.debug: loss_calc_start = time.time()
                loss = self.criterion(outputs, y_batch.double())  # Because converting to double type
                if self.debug: loss_calc_time = time.time() - loss_calc_start
                
                if self.debug: backward_start = time.time()
                loss.backward()
                if self.debug: backward_time = time.time() - backward_start
                
                if self.debug: param_update_start = time.time()
                self.optimizer.step()
                if self.debug: param_update_time = time.time() - param_update_start
                
            if self.debug:  
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.20f}, Time: {time.time()-epoch_start:.4f}s")
            # print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.20f}, Time: {time.time()-epoch_start:.4f}s")
        print("Train completed")
        
    def predict(self, X_test):
        self.model.eval()
        X_test = X_test.to(self.device)
        with torch.no_grad():
            predictions = self.model(X_test.double())  # Because converting to double type
        return predictions.cpu()

if __name__ == "__main__":
    device = torch.device("cpu") 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Because detecting if CUDA is available
    numStock = 481
    model = CNNLSTM(numStock).to(device)  # Because moving the model to CUDA device
    input_tensor = torch.rand((numStock, 1, 1300, 52)).to(device).double()  # Because generating input tensor, moving to CUDA device, and converting to double type
    lstm_input_tensor = torch.rand((numStock, 1300, 52)).to(device).double()  # Because generating LSTM input tensor, moving to CUDA device, and converting to double type
    
    # output_tensor = model(input_tensor)  # Because the input tensor is already converted to double type
    # print("Output shape:", output_tensor.shape)
    # print(f"MLPBlock: {count_parameters(MLPBlock(numStock))}".rjust(30))
    # print(f"LSTMBlock: {count_parameters(LSTMBlock(numStock))}".rjust(30))
    
    # Train the model with the training data
    stock_prediction_model = NNPredictionModel(numStock, learning_rate=0.002, epochs=1000, batch_size=481)
    # stock_prediction_model.model = nn.DataParallel(stock_prediction_model.model)  # Because wrapping the model in DataParallel
    stock_prediction_model.model.to(device)  # Because sending the model to the device
    input_tensor_y = torch.rand((numStock, 1, 1300, 1)).to(device).double()  # Because generating target tensor, moving to CUDA device, and converting to double type
    print(f'input tenshor X shape: {input_tensor.shape}')
    print(f'input tenshor y shape: {input_tensor_y.shape}')
    stock_prediction_model.train(input_tensor, input_tensor_y)  # Because the input tensors are already converted to double type
