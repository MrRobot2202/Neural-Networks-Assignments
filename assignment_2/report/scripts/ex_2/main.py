from discriminator import Discriminator
from generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epoch_losses_discrim = []
epoch_losses_gen = []

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
os.makedirs('images', exist_ok=True)


print('==> Building model..')
gen = Generator().to(device)
discrim = Discriminator().to(device)

print(f'Using device: {device}')

# recommended values from here https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
criterion = torch.nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_discrim = optim.Adam(discrim.parameters(), lr=0.0002, betas=(0.5, 0.999))

noise_dimension = 100
num_epochs = 150

print('==> Starting training..')
for epoch in range(num_epochs):
    batch_d_losses = []
    batch_g_losses = []

    for i, (img_true, label) in enumerate(trainloader):
        batch_size = img_true.size(0)
        img_true = img_true.to(device)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # only train every second iteration because of the discriminator being too good
        if i % 2 == 0:
            optimizer_discrim.zero_grad()
            noise = torch.randn(batch_size, noise_dimension, device=device)
            img_fake = gen(noise)
            real_loss = criterion(discrim(img_true), valid)
            fake_loss = criterion(discrim(img_fake.detach()), fake)
            av_loss = (real_loss + fake_loss) / 2
            batch_d_losses.append(av_loss.item())
            av_loss.backward()
            optimizer_discrim.step()
        else:
            # add the data without training first
            with torch.no_grad():
                noise = torch.randn(batch_size, noise_dimension, device=device)
                img_fake = gen(noise)
                real_loss = criterion(discrim(img_true), valid)
                fake_loss = criterion(discrim(img_fake), fake)
                av_loss = (real_loss + fake_loss) / 2
                batch_d_losses.append(av_loss.item())

        optimizer_gen.zero_grad()
        noise = torch.randn(batch_size, noise_dimension, device=device)
        img_fake = gen(noise)
        g_loss = criterion(discrim(img_fake), torch.ones(batch_size, 1, device=device))
        batch_g_losses.append(g_loss.item())
        g_loss.backward()
        optimizer_gen.step()

    avg_d_loss = sum(batch_d_losses) / len(batch_d_losses)
    avg_g_loss = sum(batch_g_losses) / len(batch_g_losses)
    epoch_losses_discrim.append(avg_d_loss)
    epoch_losses_gen.append(avg_g_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}] D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}')

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            noise = torch.randn(36, noise_dimension, device=device)
            generated = gen(noise)
            vutils.save_image(generated, f'images/epoch_{epoch+1}.png', normalize=True, nrow=8)
            print(f'Saved images for epoch {epoch+1}')

# Save results
torch.save({
    'epoch_losses_discrim': epoch_losses_discrim,
    'epoch_losses_gen': epoch_losses_gen,
}, 'results.pth')

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10,6))
plt.plot(epochs, epoch_losses_discrim, label='Discrim Loss', linewidth=2, marker='o', markersize=2)
plt.plot(epochs, epoch_losses_gen, label='Gen Loss', linewidth=2, marker='o', markersize=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Losses per Epoch')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig("training_losses_epoch.png", dpi=150)
plt.show()
print('Done')


