import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna.visualization as ov

# 1. Define the Objective Function
def objective(trial):
    # Hyperparameters to be optimized
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 5, 15)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    # Model definition (simple CNN for MNIST)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc = nn.Linear(320, 10) # 320 = 20 * 4 * 4 (output of conv2 after pooling)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 320)
            x = self.fc(x)
            return torch.log_softmax(x, dim=1)

    model = Net()

    # Optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == '__main__':
    # 2. Create a Study and Optimize (saving to a database)
    # Use a SQLite database for persistent storage of study results
    # This allows you to resume studies or analyze results later
    study = optuna.create_study(direction='maximize', study_name='mnist_optimization',
                                storage='sqlite:///db.sqlite3', load_if_exists=True)
    
    print("Starting optimization...")
    study.optimize(objective, n_trials=50, timeout=600) # Run 50 trials or for 600 seconds

    print("\nOptimization finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 3. Plotting Results
    # Requires plotly and matplotlib to be installed:
    # pip install plotly matplotlib
    
    # Plot optimization history
    try:
        fig_history = ov.plot_optimization_history(study)
        fig_history.write_image("optimization_history.png")
        print("Optimization history plot saved to optimization_history.png")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    # Plot parameter importances
    try:
        fig_importance = ov.plot_param_importances(study)
        fig_importance.write_image("param_importances.png")
        print("Parameter importances plot saved to param_importances.png")
    except Exception as e:
        print(f"Could not generate parameter importances plot: {e}")

    # Plot slice (for 2 parameters)
    try:
        fig_slice = ov.plot_slice(study, params=['lr', 'batch_size'])
        fig_slice.write_image("slice_plot.png")
        print("Slice plot saved to slice_plot.png")
    except Exception as e:
        print(f"Could not generate slice plot: {e}")

    # 4. Saving the Best Model (Placeholder)
    # In a real scenario, you would save the model trained with the best hyperparameters
    # For example: torch.save(model.state_dict(), 'best_model.pth')
    print("\nNote: In a real application, you would save the model trained with the best hyperparameters here.")
    print("For example: torch.save(model.state_dict(), 'best_model.pth')")
