import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt

from Autoencoder import Autoencoder

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Autoencoder Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--latent', default=128, type=int, help='latent channels')
parser.add_argument('--batch_number', default=0, type=int, help='Index of the test batch to use for reconstruction visualization')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = float('inf')
start_epoch = 0

train_losses = []
val_losses = []

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
autoenc = Autoencoder(latent_channels=args.latent).to(device)

if device == 'cuda':
    autoenc = torch.nn.DataParallel(autoenc)
    cudnn.benchmark = True

checkpointPath = './checkpoint_autoenc'
os.makedirs(checkpointPath, exist_ok=True)

if args.resume:
    ckptFile = os.path.join(checkpointPath, 'autoencoder_ckpt.pth')
    if os.path.isfile(ckptFile):
        print('==> Resuming from checkpoint..')
        ckpt = torch.load(ckptFile, map_location=device)
        autoenc.load_state_dict(ckpt['autoenc_state'])
        best_loss = ckpt.get('best_loss', best_loss)
        start_epoch = ckpt.get('epoch', 0) + 1
    else:
        print('No checkpoint found at', ckptFile)

# Loss, optimizer, scheduler
criterion = nn.L1Loss()
optimizer = optim.Adam(autoenc.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    autoenc.train()
    running_loss = 0.0
    batches = 0

    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = autoenc(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

        print(f"Epoch {epoch:03d} [{batch_idx+1:>4}/{len(trainloader)}] | "
              f"lr={optimizer.param_groups[0]['lr']:.4g} | "
              f"train_loss={running_loss/batches:.6f}", flush=True)

    epoch_loss = running_loss / max(1, batches)
    train_losses.append(epoch_loss)


def validate(epoch):
    global best_loss
    autoenc.eval()
    running_loss = 0
    batches = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = autoenc(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()
            batches += 1

            print(f"Val Epoch {epoch:03d} | val_loss={running_loss/batches:.6f}", flush=True)

    epoch_loss = running_loss / max(1, batches)
    val_losses.append(epoch_loss)

    # save best and checkpoint
    if epoch_loss < best_loss:
        print('Saving best autoencoder')
        state = {
            'autoenc_state': autoenc.state_dict(),
            'epoch': epoch,
            'best_loss': epoch_loss
        }
        torch.save(state, os.path.join(checkpointPath, 'autoencoder_ckpt.pth'))
        model_to_save = autoenc.module if isinstance(autoenc, nn.DataParallel) else autoenc
        if hasattr(model_to_save, 'encoder'):
            torch.save(model_to_save.encoder.state_dict(), os.path.join(checkpointPath, 'encoder.pth'))
        else:
            torch.save(autoenc.state_dict(), os.path.join(checkpointPath, 'encoder.pth'))
        best_loss = epoch_loss


for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    validate(epoch)
    scheduler.step()

# Save training curves
torch.save({
    'train_losses': train_losses,
    'val_losses': val_losses
}, 'autoencoder_results.pth')

# visualize reconstructions and save image
def visualizeReconstructions_simple(model, dataloader, batch_index=0, numImages=10,
                                    outFile='reconstructions.png'):
    model.eval()

    imgs = None
    recons = None

    with torch.no_grad():
        # iteratin through dataloader until the target batch_index is reached
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx == batch_index:
                inputs = inputs.to(device)
                outputs = model(inputs)

                # take the first 'numImages' from this specific batch
                imgs = inputs[:numImages].cpu()
                recons = outputs[:numImages].cpu()
                break

            # if the batch_index is too high
            if batch_idx > batch_index:
                print(f"Could not find batch with index {batch_index}.")
                return

    if imgs is None or recons is None:
        print(f"Could not find batch with index {batch_index}.")
        return

    total_images = len(imgs)

    # Create grids
    grid_orig = vutils.make_grid(imgs, nrow=total_images, padding=2, normalize=False)
    grid_recon = vutils.make_grid(recons, nrow=total_images, padding=2, normalize=False)

    # put the original images row on top of the reconstructed images row
    combined = torch.cat((grid_orig, grid_recon), 1)

    plt.figure(figsize=(20, 6))
    plt.axis('off')

    # Display combined image
    plt.imshow(combined.permute(1, 2, 0).numpy())

    plt.savefig(outFile, bbox_inches='tight')
    plt.close()

model_to_use = autoenc.module if isinstance(autoenc, nn.DataParallel) else autoenc
visualizeReconstructions_simple(
    model_to_use,
    testloader,
    batch_index=args.batch_number - 1,
    numImages=10,  # first 10 images from that batch
    outFile='reconstructions.png'
)

# Plot losses
iterations = range(1, len(train_losses)+1)
plt.figure(figsize=(9,6))
plt.plot(iterations, train_losses, label='Train Loss')
plt.plot(iterations, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Reconstruction Loss')
plt.legend()
plt.savefig('autoencoder_loss.png')
plt.show()
