import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from DownstreamClassifier import DownstreamClassifier

import random
import numpy as np

# random seed for reproducibility
SEED = 13
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


def train_classifier(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100. * correct / total


def eval_classifier(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), 100. * correct / total

# FINE-TUNING TRAINING
pretrained_path = './checkpoint_autoenc/encoder.pth'

model_finetune = DownstreamClassifier(
    latent_channels=128,
    num_classes=10,
    pretrained_encoder_path=pretrained_path,
).to(device)

criterion_cls = nn.CrossEntropyLoss()
optimizer_finetune = optim.Adam(model_finetune.parameters(), lr=1e-3)

num_epochs = 50
train_accs_finetune, test_accs_finetune = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_classifier(model_finetune, trainloader, optimizer_finetune, criterion_cls)
    val_loss, val_acc = eval_classifier(model_finetune, testloader, criterion_cls)
    train_accs_finetune.append(train_acc)
    test_accs_finetune.append(val_acc)
    print(f"[Finetune] Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")


# SCRATCH TRAINING
model_scratch = DownstreamClassifier(
    latent_channels=128,
    num_classes=10,
    pretrained_encoder_path=pretrained_path,
).to(device)

# reset encoder params for scratch
model_scratch.encoder.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=1e-3)

train_accs_scratch, test_accs_scratch = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_classifier(model_scratch, trainloader, optimizer_scratch, criterion_cls)
    val_loss, val_acc = eval_classifier(model_scratch, testloader, criterion_cls)
    train_accs_scratch.append(train_acc)
    test_accs_scratch.append(val_acc)
    print(f"[Scratch] Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")

#Plotting
plt.figure(figsize=(10,6))
plt.plot(train_accs_finetune, label='Finetune Train Acc')
plt.plot(test_accs_finetune, label='Finetune Test Acc')
plt.plot(train_accs_scratch, label='Scratch Train Acc')
plt.plot(test_accs_scratch, label='Scratch Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Downstream classification: pretrained vs scratch')
plt.legend()
plt.savefig('downstream_accuracy_comparison.png')
plt.show()

#saving metrics
torch.save({
    'train_accs_finetune': train_accs_finetune,
    'test_accs_finetune': test_accs_finetune,
    'train_accs_scratch': train_accs_scratch,
    'test_accs_scratch': test_accs_scratch
}, 'downstream_results.pth')

