import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt


class SimpleLowRankLayer(nn.Module):
    # low-rank matrix to add on the original qkv-weight matrix (attention head)
    def __init__(self, original_layer, rank):
        super(SimpleLowRankLayer, self).__init__()
        assert isinstance(original_layer, nn.Linear)

        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Initialize low-rank matrices A and B with small random values
        self.A = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(self.out_features, self.rank) * 0.01)


    def forward(self, x):
        # Compute the output using the original layer
        original_output = self.original_layer(x)

        # Compute the low-rank approximated output
        # Reshape to 2D tensor: [batch_size * num_patches, in_features]
        x_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        # Perform linear operations
        low_rank_output = F.linear(x_2d, self.B @ self.A)
        # Reshape back to 3D tensor: [batch_size, num_patches, out_features]
        low_rank_output = low_rank_output.view(x_shape[0], x_shape[1], self.out_features)

        # Combine the outputs (you can also combine them in a more complex way if needed)
        combined_output = original_output + low_rank_output

        return combined_output

    def freeze_original_weights(self):
        # freeze the original weights (requires_grad=False)
        for param in self.original_layer.parameters():
            param.requires_grad = False


class LowRankTransformerWrapper(nn.Module):
    def __init__(self, pretrained_model, rank):
        super(LowRankTransformerWrapper, self).__init__()
        self.pretrained_model = pretrained_model

        # the linear layer of blocks.xx.attn.qkv is extracted to be replaced for each block
        for block in self.pretrained_model.blocks:
            qkv = block.attn.qkv
            block.attn.qkv = SimpleLowRankLayer(qkv, rank)
    def forward(self, x):
        return self.pretrained_model(x)


class AttentionWithLoRA(nn.Module):
    def __init__(self, dim, heads, rank):
        super(AttentionWithLoRA, self).__init__()

        self.W_q_original = nn.Linear(dim, dim)
        self.W_k_original = nn.Linear(dim, dim)
        self.W_v_original = nn.Linear(dim, dim)

        self.W_q_low_rank = SimpleLowRankLayer(self.W_q_original, rank)
        self.W_k_low_rank = SimpleLowRankLayer(self.W_k_original, rank)
        self.W_v_low_rank = SimpleLowRankLayer(self.W_v_original, rank)

    def forward(self, x):
        Q = self.W_q_low_rank(self.W_q_original(x))
        K = self.W_k_low_rank(self.W_k_original(x))
        V = self.W_v_low_rank(self.W_v_original(x))

        # ... (rest of the attention mechanism)


# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
path = 1
train_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/train", transform=transform)
test_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize ViT model
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)

# Wrap the model with LowRankTransformerWrapper
rank = 16  # Choose rank
model = LowRankTransformerWrapper(model, rank)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Fine-tuning the model
num_epochs = 1
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

