import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class LowRankLayer(nn.Module):
    # Approximate original weight with a low-rank matrix
    # These new matrix contain the weights to be learned
    # Method: SVD followed by matrix reduction by a given rank
    def __init__(self, original_layer, rank):
        super(LowRankLayer, self).__init__()
        assert isinstance(original_layer, nn.Linear)

        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Perform SVD on the original MLP layers (i.e. the weight matrix for these layers)
        u, s, v = torch.svd(original_layer.weight.data)

        # extracting the first columns (defined by the rank number) of the SVD
        self.u_approx = nn.Parameter(u[:, :self.rank])
        self.s_approx = nn.Parameter(s[:self.rank])
        self.v_approx = nn.Parameter(v[:, :self.rank])

    def forward(self, x):
        # 1. construct the approximated weight matrix by multiplying u_approx, s_approx, and v_approx
        # 2. perform the linear operation with the feature map using this approximated weights
        # now these are the weights to be learned during fine-tuning
        approx_weight = self.u_approx @ torch.diag(self.s_approx) @ self.v_approx.T
        return F.linear(x, approx_weight)

class LowRankWrapper(nn.Module):
    def __init__(self, pretrained_model, rank_dict):
        super(LowRankWrapper, self).__init__()
        self.pretrained_model = pretrained_model
        self.rank_dict = rank_dict

        for name, module in pretrained_model.named_children():
            if name in rank_dict:
                setattr(pretrained_model, name, LowRankLayer(module, rank_dict[name]))

    def forward(self, x):
        return self.pretrained_model(x)

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/train", transform=transform)
test_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize ViT model
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)

# Wrap the model with LowRankWrapper
rank_dict = {'head': 50}
model = LowRankWrapper(model, rank_dict)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Fine-tuning the model
num_epochs = 2
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.3f}")

    # Model Evaluation
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
    print(f'Accuracy: {accuracy:.2f}%')
    accuracy_list.append(accuracy)
    scheduler.step()

# Save model
# torch.save(model.state_dict(), "model_tuned.pt")

# Plot accuracy
plt.plot(accuracy_list)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.savefig('model_accuracy.png')

'''
# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
'''

