import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 16)
        self.fc_mean = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

# 2. Sampling Layer
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

# 3. Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64 * 7 * 7)
        self.conv_t1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.conv_t1(x))
        x = torch.sigmoid(self.conv_t2(x))
        return x

# 4. VAE Model
class VAE(nn.Module):
    def __init__(self, encoder, decoder, sampling_layer):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling_layer = sampling_layer

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling_layer(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    def vae_loss(self, reconstruction, x, z_mean, z_log_var):
        reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return reconstruction_loss + kl_loss

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 2
    epochs = 1
    batch_size = 128
    learning_rate = 1e-3

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST Dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize VAE
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    sampling_layer = Sampling().to(device)
    vae = VAE(encoder, decoder, sampling_layer).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # Training the VAE
    print("\nTraining VAE...")
    for epoch in range(epochs):
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device).view(-1, 1, 28, 28)

            # Forward pass
            reconstruction, z_mean, z_log_var = vae(x)
            loss = vae.vae_loss(reconstruction, x, z_mean, z_log_var)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Example of generating new digits
    print("\nGenerating new digits...")
    with torch.no_grad():
        # Generate random latent vectors
        random_latent_vectors = torch.randn(10, latent_dim).to(device)
        generated_images = vae.decoder(random_latent_vectors).cpu()

        # Plotting function
        def plot_digits(instances, images_per_row=10):
            size = instances.shape[2]
            images_per_row = min(len(instances), images_per_row)
            images = [instance.reshape(size, size) for instance in instances]
            n_rows = (len(instances) - 1) // images_per_row + 1
            row_images = []
            n_empty = n_rows * images_per_row - len(instances)
            images.append(np.zeros((size, size * n_empty)))
            for row in range(n_rows):
                r_images = images[row * images_per_row : (row + 1) * images_per_row]
                row_images.append(np.concatenate(r_images, axis=1))
            image = np.concatenate(row_images, axis=0)
            plt.imshow(image, cmap="binary")
            plt.axis("off")
            plt.show()

        print("\nGenerated Images:")
        plot_digits(generated_images)