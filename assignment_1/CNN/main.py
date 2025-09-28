'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import math

from model import *
from model.CNN import CNN


import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = CNN()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(f"Epoch {epoch:03d} [{batch_idx + 1:>4}/{len(trainloader)}] |"
              f"lr={optimizer.param_groups[0]['lr']:.4g} | "
              f"train_loss={train_loss:.4f} | train_acc={100. * correct / total:.3f}",
              flush=True)

    train_losses.append(train_loss / len(trainloader))
    train_accuracies.append(100 * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(f"Epoch {epoch:03d} |"
                  f"lr={optimizer.param_groups[0]['lr']:.4g} | "
                  f"test_loss={test_loss:.4f} | test_acc={100. * correct / total:.3f}",
                  flush=True)

    test_losses.append(test_loss / len(testloader))
    test_accuracies.append(100 * correct / total)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def visualize_feature_maps(net, dataloader):
    print('\n==> Visualizing Feature Maps..')
    net.eval()

    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    input_tensor = images[0:1].to(device)

    with torch.no_grad():
        model_to_use = net.module if isinstance(net, nn.DataParallel) else net
        outputs, feature_maps = model_to_use(input_tensor, get_features=True)

    blocks_to_show = [0, 1, 2]
    channels_to_show = [0, 10, 20]

    plt.figure(figsize=(12, 12))

    for i, block_index in enumerate(blocks_to_show):
        feature_map_tensor = feature_maps[block_index].cpu().squeeze(0)  # [C, H, W]
        channels_count, H, W = feature_map_tensor.size()

        print(f"Feature Map Block {block_index + 1} size: {channels_count}x{H}x{W}")

        for j, channel_index in enumerate(channels_to_show):
            if channel_index < channels_count:
                plot_index = i * 3 + j + 1
                plt.subplot(3, 3, plot_index)

                feature_2d = feature_map_tensor[channel_index].numpy()

                plt.imshow(feature_2d, cmap='viridis')
                plt.title(f'Block {block_index + 1}, Ch {channel_index}\nSize: {H}x{W}')
                plt.axis('off')

    plt.tight_layout()
    plt.savefig("feature_maps_visualization.png")
    plt.show()


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()

import torch

# I needed to save the results so I dont have to train the model again, I forgot the first time.....
torch.save({
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies
}, 'results.pth')

visualize_feature_maps(net, testloader)

#plotting stuff
iterations = range(1, len(train_losses)+1)

plt.figure(figsize=(9,6))
plt.plot(iterations, train_losses, label= 'Train Losses')
plt.plot(iterations, test_losses, label= 'Test Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Test Losses')
plt.legend()
plt.show()
plt.savefig("Training & Test Losses")

plt.figure(figsize=(9,6))
plt.plot(iterations, train_accuracies, label='Train Accuracy')
plt.plot(iterations, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy in %')
plt.title('Training & Test Accuracies')
plt.legend()
plt.show()
plt.savefig("Training & Test Accuracies")