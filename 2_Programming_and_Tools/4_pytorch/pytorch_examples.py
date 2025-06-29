# PyTorch Examples

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Tensor Operations
print("--- Tensor Operations ---")
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("Original Tensor:\n", x)
print("Tensor + 1:\n", x + 1)
print("Matrix Multiplication (x, x.T):\n", torch.matmul(x, x.T))

# 2. Simple Neural Network
print("\n--- Simple Neural Network ---")
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNN()
print("Model Architecture:\n", model)

# Dummy data
X = torch.randn(10, 2) # 10 samples, 2 features
y = torch.randn(10, 1) # 10 samples, 1 target

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (simplified)
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Make a prediction
new_data = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
with torch.no_grad():
    prediction = model(new_data)
print("Prediction for [0.5, 0.8]:", prediction.item())
