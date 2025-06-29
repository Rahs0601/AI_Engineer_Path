import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Generation
def generate_sine_wave(timesteps, amplitude, frequency, noise_level=0.1):
    time = np.arange(timesteps)
    data = amplitude * np.sin(2 * np.pi * frequency * time) + noise_level * np.random.randn(timesteps)
    return data

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 2. Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Get the output from the last time step
        return out

if __name__ == "__main__":
    # Hyperparameters
    timesteps = 1000
    amplitude = 10
    frequency = 0.05
    seq_length = 10
    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    learning_rate = 0.001
    epochs = 100
    train_split = 0.8

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data
    data = generate_sine_wave(timesteps, amplitude, frequency)
    
    # Create sequences
    X, y = create_sequences(data, seq_length)
    
    # Reshape X for LSTM input (batch_size, seq_length, input_size)
    X = X.reshape(-1, seq_length, input_size)

    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().unsqueeze(1) # Add feature dimension for output

    # Split data into training and testing sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model, loss, and optimizer
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nTraining LSTM model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train.to(device)).cpu().numpy()
        test_predict = model(X_test.to(device)).cpu().numpy()

    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(data)
    train_predict_plot[:] = np.nan
    train_predict_plot[seq_length:len(train_predict) + seq_length] = train_predict.flatten()

    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(data)
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict) + seq_length:] = test_predict.flatten()

    # Plot original data and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original Data')
    plt.plot(train_predict_plot, label='Train Prediction')
    plt.plot(test_predict_plot, label='Test Prediction')
    plt.title('Time Series Forecasting with LSTM')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
