import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(192, 256, batch_first=True)
        self.fc = nn.Linear(256, 26)
    def forward(self, x):
        out, _ = self.lstm(x)  # Output will have shape (batch_size, 100, 64)
        out = out[:, -1, :]  # Now out has shape (batch_size, 64)
        out = self.fc(out)  # Now out has shape (batch_size, 10)
        return out

# Define the inception block
class InceptionBlock(nn.Module):
    def __init__(self):
        super(InceptionBlock, self).__init__()
        
        self.subblock1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )
        
        self.subblock2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        )
        
        self.subblock3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, x):
        x1 = self.subblock1(x)
        x2 = self.subblock2(x)
        x3 = self.subblock3(x)
        stacked = torch.stack((x1, x2, x3), dim=4)
        permuted = stacked.permute(0, 2, 1, 3, 4)
        reshaped = permuted.reshape(1, 1300, -1)
        return reshaped


# Define the main model
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0)),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(1, 0))
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0)),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(1, 0))
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10), stride=(1, 10))
        )
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        return x
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv = ConvBlock()
        self.inception = InceptionBlock()
        self.lstm_block = LSTMBlock()
    def forward(self, x):
        x = self.conv(x)
        x = self.inception(x)
        x = self.lstm_block(x)
        return x
if __name__=="__main__":
    # Create an instance of the model
    model = CNNLSTM()
    # Create a dummy input tensor
    input_tensor = torch.rand((1, 1, 1300, 52))
    # Forward pass
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)
