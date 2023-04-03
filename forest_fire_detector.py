#Importing the Libraries
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

#Function to compute the Mean and Standard Deviation of the Dataset
def calculate_dataset_mean_std(dataset, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),  # resize images to a common size
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = 0.0
    var = 0.0
    num_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        var += images.var(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    var /= num_samples
    std = torch.sqrt(var)
    return mean, std

# Loading the Dataset
def load_dataset(data_dir, mean, std):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

data_dir = 'Dataset'
unnormalized_dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
mean, std = calculate_dataset_mean_std(unnormalized_dataset)
print(f"Mean: {mean}, Standard Deviation: {std}")

dataset = load_dataset(data_dir, mean, std)

# Splitting the dataset into training, validation and testing sets
train_size = int(0.7 * len(dataset))
validation_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Loading the pre-trained model (Transfer Learning)
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training and validating the model
num_epochs = 10
best_val_accuracy = 0
best_model = None
model_save_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, (inputs, labels) in progress_bar:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

    # Validating the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Accuracy on validation set for epoch {epoch + 1}: {np.round(val_accuracy, 3)}%')
    
    # Saving the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = deepcopy(model)
        torch.save(best_model.state_dict(), model_save_path)
        print("Model updated")

# Testing the best model
best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy on test set: {np.round(100 * correct / total, 3)}%')