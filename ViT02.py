import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Image preprocessing for vit_tiny_patch16_224 and Normalizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageWoof training and test data from path and transform the images
train_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/train", transform=transform)
test_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize ViT model
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Set initial high lr
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # Reduce lr by a factor of 0.7 between epochs


# Fine-tune model
num_epochs = 5
accuracy_list = []

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"epoch [{epoch+1}/{num_epochs}], step [{batch_idx+1}/{len(train_loader)}], loss: {loss.item():.3f}")


    # model evaluation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(target.numpy())

    accuracy = (100 * correct / total)
    print(f'accuracy: {accuracy:.2f}%')
    accuracy_list.append(accuracy)
    scheduler.step()  # Reduce the lr to next epoch

# Save model
torch.save(model.state_dict(), "model_tuned.pt")

# Plot accuracy
plt.plot(accuracy_list)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.savefig('model_accuracy.png')

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('predicted')
plt.ylabel('truth')
plt.title('confusion matrix')
plt.savefig('confusion_matrix.png')

