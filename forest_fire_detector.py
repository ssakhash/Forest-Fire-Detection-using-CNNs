import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Custom dataset class that inherits from torchvision.datasets.ImageFolder
class CustomDataset(datasets.ImageFolder):
    def __init__(self, data_dir, mean=None, std=None, *args, **kwargs):
        # Apply different transforms based on whether mean and std are provided
        if mean is not None and std is not None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        super().__init__(data_dir, transform=transform, *args, **kwargs)

    @staticmethod
    # Function to compute the mean and standard deviation of the dataset
    def calculate_mean_std(dataset, batch_size=64):
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

# Image classifier class to handle training, validation, and testing
class ImageClassifier:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    # Function to train the model for one epoch
    def train_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in progress_bar:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

    # Function to validate the model
    def validate(self, validation_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # Function to test the model
    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    # Function to calculate metrics and plot the ROC curve
    def calculate_metrics_and_plot_roc(self, test_loader):
        self.model.eval()

        y_true = []
        y_pred = []
        y_score = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_score.extend(probabilities[:, 1].cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"AUC: {auc}")

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

def main():
    data_dir = 'Dataset'
    # Load dataset without normalization to compute mean and std
    unnormalized_dataset = CustomDataset(data_dir)
    mean, std = CustomDataset.calculate_mean_std(unnormalized_dataset)
    print(f"Mean: {mean}, Standard Deviation: {std}")

    # Load normalized dataset
    dataset = CustomDataset(data_dir, mean, std)

    # Splitting the dataset into training, validation, and testing sets
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

    # Initialize the image classifier
    num_epochs = 10
    best_val_accuracy = 0
    best_model = None
    model_save_path = "best_model.pth"

    classifier = ImageClassifier(model, criterion, optimizer)

    for epoch in range(num_epochs):
        classifier.train_epoch(train_loader, epoch, num_epochs)

        # Validating the model
        val_accuracy = classifier.validate(validation_loader)
        print(f'Accuracy on validation set for epoch {epoch + 1}: {np.round(val_accuracy, 3)}%')

        # Saving the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(classifier.model)
            torch.save(best_model.state_dict(), model_save_path)
            print("Model updated")

    # Testing the best model
    best_classifier = ImageClassifier(best_model, criterion, optimizer)
    test_accuracy = best_classifier.test(test_loader)
    print(f'Accuracy on test set: {np.round(test_accuracy, 3)}%')

    # Calculate metrics and plot ROC curve
    best_classifier.calculate_metrics_and_plot_roc(test_loader)

if __name__ == "__main__":
    main()