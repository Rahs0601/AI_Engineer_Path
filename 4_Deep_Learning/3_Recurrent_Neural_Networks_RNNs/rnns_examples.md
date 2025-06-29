# Recurrent Neural Networks (RNNs) Examples

This file provides examples of building and training simple Recurrent Neural Networks (RNNs) using PyTorch, primarily for sequence data tasks like text generation or time series prediction.

## 1. Simple RNN for Character-Level Language Modeling

RNNs are designed to recognize sequential characteristics and are particularly useful for tasks involving sequences, such as natural language processing and time series analysis.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Prepare Data
# Simple character-level dataset: predict the next character in a sequence
text = "hello world"
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

input_size = len(chars) # Number of unique characters
hidden_size = 50
output_size = len(chars)
learning_rate = 0.01
num_epochs = 100

# Convert text to one-hot encoded tensors
def text_to_one_hot(text_seq, char_to_idx, input_size):
    tensor = torch.zeros(len(text_seq), 1, input_size)
    for i, char in enumerate(text_seq):
        tensor[i][0][char_to_idx[char]] = 1
    return tensor

def target_to_long(text_seq, char_to_idx):
    return torch.LongTensor([char_to_idx[char] for char in text_seq])

# Create input and target sequences
input_seq = text_to_one_hot(text[:-1], char_to_idx, input_size)
target_seq = target_to_long(text[1:], char_to_idx)

# 2. Define the RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) # batch_first=True for (batch, seq, feature)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden_state):
        output, hidden = self.rnn(input_tensor, hidden_state)
        output = self.fc(output.reshape(-1, self.hidden_size)) # Reshape for linear layer
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size) # (num_layers * num_directions, batch_size, hidden_size)

model = RNN(input_size, hidden_size, output_size)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3. Training the Model
print("--- Training Simple RNN for Character Prediction ---")
for epoch in range(num_epochs):
    hidden = model.init_hidden()
    optimizer.zero_grad()

    # Forward pass
    outputs, hidden = model(input_seq, hidden)
    loss = criterion(outputs, target_seq)

    # Backward and optimize
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. Generate Text (Inference)
print("\n--- Generating Text with Trained RNN ---")
start_char = 'h'
predicted_text = start_char

with torch.no_grad():
    hidden = model.init_hidden()
    input_char = text_to_one_hot(start_char, char_to_idx, input_size)

    for _ in range(len(text) - 1):
        output, hidden = model(input_char, hidden)
        # Get the most likely next character
        top_value, top_index = output.topk(1)
        predicted_char = idx_to_char[top_index.item()]
        predicted_text += predicted_char

        # Use the predicted character as the next input
        input_char = text_to_one_hot(predicted_char, char_to_idx, input_size)

print(f"Generated text: {predicted_text}")
```

## 2. Long Short-Term Memory (LSTM) Network

LSTMs are a special kind of RNN, capable of learning long-term dependencies. They were introduced to address the vanishing/exploding gradient problems that can be encountered when training traditional RNNs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data preparation (same as simple RNN example)
text = "hello world"
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

input_size = len(chars)
hidden_size = 50
output_size = len(chars)
learning_rate = 0.01
num_epochs = 100

def text_to_one_hot(text_seq, char_to_idx, input_size):
    tensor = torch.zeros(len(text_seq), 1, input_size)
    for i, char in enumerate(text_seq):
        tensor[i][0][char_to_idx[char]] = 1
    return tensor

def target_to_long(text_seq, char_to_idx):
    return torch.LongTensor([char_to_idx[char] for char in text_seq])

input_seq = text_to_one_hot(text[:-1], char_to_idx, input_size)
target_seq = target_to_long(text[1:], char_to_idx)

# Define the LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) # batch_first=True for (batch, seq, feature)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden_and_cell_state):
        output, (hidden, cell) = self.lstm(input_tensor, hidden_and_cell_state)
        output = self.fc(output.reshape(-1, self.hidden_size))
        return output, (hidden, cell)

    def init_hidden_and_cell(self):
        return (torch.zeros(1, 1, self.hidden_size), # hidden state
                torch.zeros(1, 1, self.hidden_size)) # cell state

model_lstm = LSTM(input_size, hidden_size, output_size)

# Loss and Optimizer
criterion_lstm = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learning_rate)

# Training the LSTM Model
print("\n--- Training LSTM for Character Prediction ---")
for epoch in range(num_epochs):
    hidden_and_cell = model_lstm.init_hidden_and_cell()
    optimizer_lstm.zero_grad()

    outputs_lstm, hidden_and_cell = model_lstm(input_seq, hidden_and_cell)
    loss_lstm = criterion_lstm(outputs_lstm, target_seq)

    loss_lstm.backward()
    optimizer_lstm.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_lstm.item():.4f}')

# Generate Text with LSTM (Inference)
print("\n--- Generating Text with Trained LSTM ---")
start_char_lstm = 'h'
predicted_text_lstm = start_char_lstm

with torch.no_grad():
    hidden_and_cell = model_lstm.init_hidden_and_cell()
    input_char_lstm = text_to_one_hot(start_char_lstm, char_to_idx, input_size)

    for _ in range(len(text) - 1):
        output_lstm, hidden_and_cell = model_lstm(input_char_lstm, hidden_and_cell)
        top_value_lstm, top_index_lstm = output_lstm.topk(1)
        predicted_char_lstm = idx_to_char[top_index_lstm.item()]
        predicted_text_lstm += predicted_char_lstm

        input_char_lstm = text_to_one_hot(predicted_char_lstm, char_to_idx, input_size)

print(f"Generated text: {predicted_text_lstm}")
```
