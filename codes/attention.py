import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(attention_size, hidden_size))
        nn.init.xavier_uniform_(self.attention_weights.data, gain=1.414)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size)
        attention_scores = torch.matmul(x, self.attention_weights.t())
        attention_scores = torch.tanh(attention_scores)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        weighted_x = torch.matmul(attention_weights.transpose(1, 2), x)
        return weighted_x.sum(dim=1), attention_weights

class AttentionModel(nn.Module):
    def __init__(self, num_features, hidden_size, attention_size):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU()
        )
        self.attention = Attention(hidden_size, attention_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        batch_size, sequence_length, _ = x.shape
        mlp_out = self.mlp(x.view(-1, x.size(-1)))  # Shape: (batch_size*sequence_length, hidden_size)
        mlp_out = mlp_out.view(batch_size, sequence_length, self.hidden_size)  # Shape: (batch_size, sequence_length, hidden_size)
        attn_output, attn_weights = self.attention(mlp_out)
        output = self.fc(attn_output)
        return output, attn_weights

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()  # Assuming a regression task
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, train_loader, num_epochs, writer):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output, _ = self.model(X_batch)
                # Reshape y_batch to match the output shape
                y_batch = y_batch[:, -1, :]  # Take the last element in the sequence
                output = output.view(-1, 1)
                y_batch = y_batch.view(-1, 1)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
                print(f'[Epoch {epoch+1}, Batch {i+1}] loss: {epoch_loss / 10:.3f}')
                writer.add_scalar('training loss', epoch_loss / 10, epoch * len(train_loader) + i)
                epoch_loss = 0.0

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            output, attn_weights = self.model(X)
        return output, attn_weights

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_stocks = 469
    train_days = 50
    bin_size = 26
    batch_size = 128  # Set batch size to a reasonable number
    num_epochs = 10
    
    # Define hyperparameters
    num_features = 52  # number of features
    hidden_size = 128  # size of MLP hidden state
    attention_size = 64  # size of attention mechanism

    # Create the model
    model = AttentionModel(num_features, hidden_size, attention_size).to(device)
    trainer = ModelTrainer(model)

    # Example input
    X_train = torch.randn((num_stocks * bin_size, train_days, num_features)).to(device)  # (num_stocks*bin_size, sequence_length, num_features)

    # Target values
    y_train = torch.randn((num_stocks * bin_size, train_days, 1)).to(device)  # (num_stocks*bin_size, sequence_length, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/attention_model_experiment')

    # Train the model
    trainer.train(train_loader, num_epochs, writer)

    # Predict on new data
    X_test = torch.randn((num_stocks * bin_size, train_days, num_features)).to(device)  # (num_stocks*bin_size, sequence_length, num_features)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    predictions, attn_weights = trainer.predict(X_test)
    print(predictions.shape)  # Should be (num_stocks*bin_size, 1)
    print(attn_weights.shape)  # Should be (num_stocks*bin_size, sequence_length, attention_size)

    # Close the TensorBoard writer
    writer.close()
