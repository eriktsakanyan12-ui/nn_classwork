import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from models import SimpleNN
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train = False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)

sample_image, sample_label = train_dataset[0]
print("Dimension of a single data sample:", sample_image.shape)


model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
 
def train_one_epoch(loader, model):
    avg_loss = []
    for x, y in loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        avg_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(avg_loss)

def train(model, loader, epochs = 5):
    loss_l = []
    for epoch in range(epochs):
        loss = train_one_epoch(loader, model)
        loss_l.append(loss)
        print(f"Epoch {epoch} - Loss: {loss:.2f}")
        PATH = f'.\model_checkpoints\mnist_simplenn_weights_epoch{epoch}.pth'
        torch.save(model.state_dict(), PATH)
    return loss_l
        

if __name__ == '__main__':
    print("Starting training")
    loss = train(model, train_loader)
    print("Training finished")