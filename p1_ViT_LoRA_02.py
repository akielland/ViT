import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import litdata
import matplotlib.pyplot as plt
from time import process_time


# Specify data folder
datapath = '/projects/ec232/data/'
t1_start = process_time()


class LowRankLayer(nn.Module):
    # Approximate original weight with a low-rank matrix
    # These new matrix contain the weights to be learned
    # Method: SVD followed by matrix reduction by a given rank
    def __init__(self, original_layer, rank):
        super(LowRankLayer, self).__init__()
        assert isinstance(original_layer, nn.Linear)

        self.rank = rank
        # self.in_features = original_layer.in_features
        # self.out_features = original_layer.out_features
        self.original_layer = original_layer  # Store a reference to the original layer

        # Perform SVD on the original MLP layers (i.e. the weight matrix for these layers)
        # u, s, v = torch.svd(original_layer.weight.data)
        u, s, v = torch.svd(self.original_layer.weight.data)

        # extracting the first columns (defined by the rank number) of the SVD
        self.u_approx = nn.Parameter(u[:, :self.rank])
        self.s_approx = nn.Parameter(s[:self.rank])
        self.v_approx = nn.Parameter(v[:, :self.rank])

    def forward(self, x):
        # Construct the approximated weight matrix by multiplying u_approx, s_approx, and v_approx
        approx_weight = self.u_approx @ torch.diag(self.s_approx) @ self.v_approx.T

        # Use the original weights as well, but with the low-rank matrix added as a delta
        combined_weights = self.original_layer.weight + approx_weight
        return F.linear(x, combined_weights)

    def freeze_original_weights(self):
        # freeze the original weights (requires_grad=False)
        self.original_layer.weight.requires_grad = False
    
    
class LowRankHeadWrapper(nn.Module):
    # Wrap around the pre-trained model to substitute its head with a low-rank approximation
    # creates a new object that has the modified head
    def __init__(self, pretrained_model, rank):
        super(LowRankHeadWrapper, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.head = LowRankLayer(self.pretrained_model.head, rank)

    def forward(self, x):
        return self.pretrained_model(x)


class ToRGBTensor:
    # code from Marius converting gray scale images to 3 times the image to imitating RGB image dimesion
    def __call__(self, x):
        return transforms.functional.to_tensor(x).expand(3, -1, -1)

# 1. Image preprocessing for vit_tiny_patch16_224 and Normalizing
# 2. Create tuple for image and class...
preprocess1 = (transforms.Compose([                        # Handles processing of the .jpg image
        transforms.Resize((224, 224)),
        ToRGBTensor(),                                     # Convert from PIL image to torch.Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image to correct mean/std.
    ]),
    nn.Identity(),                     # Handles proc. of .cls file (just an int).
)

train_dataset = litdata.LITDataset('ImageWoof', datapath).map_tuple(*preprocess1)
test_dataset = litdata.LITDataset('ImageWoof', datapath, train=False).map_tuple(*preprocess1)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize ViT model
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)

# Wrap the model with LowRankHeadWrapper
rank = 50
model = LowRankHeadWrapper(model, rank)
# Freeze the original weights of the head
# model.freeze_original_weights()
model.pretrained_model.head.freeze_original_weights()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Fine-tuning the model
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
torch.save(model,'output/lora_model_2_50.pth')

# Plot accuracy
plt.plot(accuracy_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.savefig('output/model_accuracy_LoRA_2_50.png')

t1_stop = process_time()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
