import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from get_results import BIN_SIZE, TRAIN_DAYS
# from trainPred import BIN_SIZE, TRAIN_DAYS
from torch.utils.tensorboard import SummaryWriter

# # Define hyperparameters
# bin_size = 26
# train_days = 50 
bin_size = BIN_SIZE
train_days = TRAIN_DAYS

class LSTMBlock(nn.Module):
    def __init__(self,numStock):
        super(LSTMBlock, self).__init__()
        self.numStock = numStock
        self.lstm = nn.LSTM(12,bin_size,num_layers=1,batch_first=True)  # TODO reduce 192 to lower !!!
        # self.lstm = nn.LSTM(26,bin_size,num_layers=1,batch_first=True)  # TODO reduce 192 to lower !!!
        # self.lstm = nn.LSTM(52,bin_size,num_layers=1,batch_first=True)  # TODO reduce 192 to lower !!!
        
        self.fc1 = nn.Linear(bin_size, 1) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)  # Output torch.Size([483, 1300, 26])
        # out = out[:, -1, :]  # '''I guess it should be used to get the last predcited bin value'''
        self.sigmoid(out)  # Activation function
        out = self.fc1(out)  # Now out has shape torch.Size([483, 1300, 1])
        # self.sigmoid(out)  # Activation function
        # out = self.fc2(out)  # Now out has shape (batch_size, 10)
        out = out.reshape(self.numStock,train_days*bin_size,1) # result is torch.Size([483, 1300, 1])
        return out
    # def forward(self, x):
    #     out, _ = self.lstm(x)  # Output will have shape (batch_size, 100, 64)
    #     # out = out[:, -1, :]  # Now out has shape (batch_size, 64)
    #     self.sigmoid(out)  # Activation function
    #     out = self.fc1(out)  # Now out has shape (batch_size, 10)
    #     # self.sigmoid(out)  # Activation function
    #     # out = self.fc2(out)  # Now out has shape (batch_size, 10)
    #     out = out.reshape(1,train_days*bin_size*self.numStock,1)
    #     return out


# Define the inception block
class InceptionBlock(nn.Module):
    def __init__(self,numStock):
        super(InceptionBlock, self).__init__()
        self.numStock=numStock


        self.subblock1 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  # No padding needed for 1x1 conv
            nn.Conv2d(4, 4, kernel_size=(5 * numStock, 1), stride=(1, 1), padding=((5 * numStock - 1) // 2, 0)),  # Padding to keep height the same
            nn.Conv2d(4, 4, kernel_size=(3 * numStock, 1), stride=(1, 1), padding=((3 * numStock - 1) // 2, 0))  # Padding to keep height the same
        )

        self.subblock2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1)) ,
            nn.Conv2d(4, 4, kernel_size=(4*numStock, 1), padding=((4 * numStock - 1) // 2+1, 0)) ,
            nn.Conv2d(4, 4, kernel_size=(2*numStock, 1), padding=((2 * numStock - 1) // 2, 0)) ,

        )

        self.subblock3 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) ,
            nn.Conv2d(4, 4, kernel_size=(bin_size*numStock, 1),padding=((bin_size * numStock - 1) // 2+1,0)) ,
            nn.Conv2d(4, 4, kernel_size=(2*numStock, 1),padding=((2 * numStock - 1) // 2,0)) ,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(36, 26),
            nn.ReLU(),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(32, 52),
        #     nn.ReLU(),
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4, 1),
        #     nn.ReLU(), 
        # )

    def forward(self, x_input):
        output = x_input.reshape(self.numStock, train_days*bin_size, 12)
        # output = self.fc2(x)
        return output
    
    
    # def forward(self, x):
    #     x1 = self.subblock1(x)
    #     x2 = self.subblock2(x)
    #     x3 = self.subblock3(x) #TODO
    #     # print(x.shape,x1.shape,'\n',x3.shape)
    #     # print(x2.shape,'\n',x3.shape)
    #     # stacked = torch.stack((x1, x3), dim=4)
    #     # breakpoint()
    #     stacked = torch.stack((x1, x2, x3),dim=0)
    #     # stacked = torch.stack((x2, x3), dim=4)
    #     # permuted = stacked.permute(0, 2, 1, 3, 4)
    #     # reshaped = stacked.reshape(1, train_days*bin_size*self.numStock, 24)
    #     reshaped = stacked.reshape(self.numStock, train_days*bin_size, 36)
    #     # output = self.fc(24,52)
    #     output = self.fc(reshaped)
    #     return output
    #     # return reshaped
    #     #'''Output reshaped shape: torch.Size([1, 1274, 192])'''



# Define the main model
class ConvBlock(nn.Module):
    def __init__(self,numStock):
        super(ConvBlock, self).__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 32)) ,
            nn.Conv2d(4, 4, kernel_size=(4*numStock, 1), padding=(2*numStock-1, 0)) ,
            nn.Conv2d(4, 4, kernel_size=(4*numStock, 1), padding=(2*numStock, 0)) 
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(1, 16)) ,
            nn.Conv2d(4, 4, kernel_size=(4*numStock, 1), padding=(2*numStock-1, 0)) ,
            nn.Conv2d(4, 4, kernel_size=(4*numStock, 1), padding=(2*numStock, 0)) 
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(1, 4)) 
        )

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, numStock):
        super().__init__()
        self.numStock = numStock
        self.fc = nn.Sequential(
            nn.Linear(52, 130) ,  # Input layer: 52 input features, 128 output features
            nn.ReLU(),  # Activation function
            nn.Linear(130, 52) ,  # Hidden layer: 128 input features, 64 output features
            nn.Sigmoid()  # Activation function to ensure output is between 0 and 1
        )
        self.fc2=nn.Sequential(
            nn.Linear(52,26),
            nn.ReLU(),
            nn.Linear(26,1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x2 = self.fc(x)  # Output shape: torch.Size([numStock, 1, train_days*bin_size, num_features])
        x3 = x2.view(self.numStock, train_days*bin_size, 52)  # torch.Size([numStock, train_days*bin_size, num_features])
        # x4 = self.fc2(x3)
        # x5 = x4.view(self.numStock, 1, train_days*bin_size, 1)  
        return x3


class CNNLSTM(nn.Module):
    def __init__(self,numStock):
        super(CNNLSTM, self).__init__()
        self.mlp = MLPBlock(numStock)
        self.conv = ConvBlock(numStock)
        self.inception = InceptionBlock(numStock)
        self.lstm_block = LSTMBlock(numStock)
    def forward(self, x_input):
        # TODO the input,x is wrong, should be with shape 1,1,1300*483,52
        # print("x_input",x_input.shape)
        x = x_input.unsqueeze(1)
        x_conv = self.conv(x) # ([7, 8, 1300, 1])
        # print("self.conv(x)",x_conv.shape)
        x = self.inception(x_conv)
        # print("self.inception(x)",x.shape)
        # x = self.mlp(x_input)
        # x = self.mlp2(x)
        # print(f"mlp_time : {time.time()-start1:.4f}s")
        x = self.lstm_block(x)
        # print(f"lstm_time: {time.time()-start2:.4f}s")
        # print("self.lstm_block",x.shape)
        return x
    
    
# Calculate number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

# Define prediction model class
class NNPredictionModel:
    def __init__(self, numStock, learning_rate=0.001, epochs=10, batch_size=32, debug=True):
        self.model = CNNLSTM(numStock)   
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Because using Adam optimizer
        self.criterion = nn.MSELoss()  # Because using Mean Squared Error loss for regression task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Because using CUDA or CPU device
        self.debug = debug
        self.writer = SummaryWriter()
        
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
                outputs = self.model(X_batch )  
                if self.debug: forward_time = time.time() - forward_start
                
                if self.debug: loss_calc_start = time.time()
                loss = self.criterion(outputs, y_batch )  
                if self.debug: loss_calc_time = time.time() - loss_calc_start
                
                if self.debug: backward_start = time.time()
                loss.backward()
                if self.debug: backward_time = time.time() - backward_start
                
                if self.debug: param_update_start = time.time()
                self.optimizer.step()
                if self.debug: param_update_time = time.time() - param_update_start
                
            # if self.debug:  
            #     print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.20f}, Time: {time.time()-epoch_start:.4f}s")
            # print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.20f}, Time: {time.time()-epoch_start:.4f}s")
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
        self.writer.close()
        print("Train completed")
        
    def predict(self, X_test):
        self.model.eval()
        X_test = X_test.to(self.device)
        with torch.no_grad():
            predictions = self.model(X_test )  
        return predictions.cpu()

if __name__ == "__main__":
    device = torch.device("cpu") 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Because detecting if CUDA is available
    # numStock = 1
    numStock = 3
    # numStock = 481
    model = CNNLSTM(numStock).to(device)  # Because moving the model to CUDA device
    input_tensor = torch.rand((numStock, 1, train_days*bin_size, 52)).to(device)   # Because generating input tensor, moving to CUDA device
    lstm_input_tensor = torch.rand((numStock, train_days*bin_size, 52)).to(device)   # Because generating LSTM input tensor, moving to CUDA device
    
    output_tensor = model(input_tensor)  
    print("Output shape:", output_tensor.shape)
    print(f"MLPBlock: {count_parameters(MLPBlock(numStock))}".rjust(30))
    print(f"LSTMBlock: {count_parameters(LSTMBlock(numStock))}".rjust(30))
    
    # Train the model with the training data
    stock_prediction_model = NNPredictionModel(numStock, learning_rate=0.001, epochs=2000, batch_size=481)
    # stock_prediction_model = NNPredictionModel(numStock, learning_rate=0.001, epochs=1000, batch_size=481)
    # stock_prediction_model.model = nn.DataParallel(stock_prediction_model.model)  # Because wrapping the model in DataParallel
    stock_prediction_model.model.to(device)  # Because sending the model to the device
    input_tensor_y = torch.rand((numStock, 1, train_days*bin_size, 1)).to(device)   # Because generating target tensor, moving to CUDA device, 
    print(f'input tenshor X shape: {input_tensor.shape}')
    print(f'input tenshor y shape: {input_tensor_y.shape}')
    stock_prediction_model.train(input_tensor, input_tensor_y)  
