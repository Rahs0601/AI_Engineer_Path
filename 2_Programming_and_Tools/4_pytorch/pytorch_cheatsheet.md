# PyTorch Cheat Sheet

## Tensors
- **Creation:** `torch.tensor([1, 2, 3])`, `torch.zeros(2, 3)`, `torch.ones(2, 3)`, `torch.rand(2, 3)`
- **Data Types:** `dtype=torch.float32`, `dtype=torch.int64`
- **Device:** `device='cuda'` or `device='cpu'`
- **Operations:**
    - Addition: `x + y` or `torch.add(x, y)`
    - Multiplication: `x * y` (element-wise) or `torch.matmul(x, y)` (matrix multiplication)
    - Reshaping: `x.view(new_shape)`, `x.reshape(new_shape)`
    - Slicing: `x[0, :]`, `x[:, 1]`
    - Item access: `x.item()` (for single-element tensors)

## Autograd (Automatic Differentiation)
- **Enabling:** `x.requires_grad_(True)`
- **Backward Pass:** `loss.backward()` (computes gradients)
- **Accessing Gradients:** `x.grad`
- **Disabling Grad:** `with torch.no_grad():` (for inference)

## Neural Networks (torch.nn)
- **Module:** `nn.Module` (base class for all neural network modules)
- **Layers:**
    - Linear: `nn.Linear(in_features, out_features)`
    - Convolutional: `nn.Conv2d(in_channels, out_channels, kernel_size)`
    - Activation: `nn.ReLU()`, `nn.Sigmoid()`, `nn.Softmax()`
    - Pooling: `nn.MaxPool2d()`, `nn.AvgPool2d()`
- **Sequential Model:** `nn.Sequential(layer1, layer2, ...)`
- **Loss Functions:**
    - MSE: `nn.MSELoss()`
    - Cross-Entropy: `nn.CrossEntropyLoss()`
- **Optimizers:**
    - SGD: `optim.SGD(model.parameters(), lr=0.01)`
    - Adam: `optim.Adam(model.parameters(), lr=0.001)`

## Training Loop
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNN()

# 2. Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Data (dummy example)
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# 4. Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad() # Clear gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

## Saving and Loading Models
- **Save:** `torch.save(model.state_dict(), 'model.pth')`
- **Load:**
    ```python
    model = SimpleNN() # Re-instantiate the model architecture
    model.load_state_dict(torch.load('model.pth'))
    model.eval() # Set to evaluation mode
    ```

## GPU Usage
- **Check availability:** `torch.cuda.is_available()`
- **Set device:** `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- **Move tensor/model to device:** `tensor.to(device)`, `model.to(device)`
