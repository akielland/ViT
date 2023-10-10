import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
import litdata
from time import process_time

#import torchvision.transforms as T


# Specify data folder
datapath = '/projects/ec232/data/'
t1_start = process_time() 
   

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

# Image preprocessing for vit_tiny_patch16_224 and Normalizing
preprocess2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageWoof training and test data from path and transform the images
# train_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/train", transform=transform)
# test_dataset = datasets.ImageFolder("/Users/anders/Documents/IN5310/project1/imagewoof2-160/val", transform=transform)

train_dataset = litdata.LITDataset('ImageWoof', datapath).map_tuple(*preprocess1)
test_dataset = litdata.LITDataset('ImageWoof', datapath, train=False).map_tuple(*preprocess1)

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
# torch.save(model.state_dict(), "full_model.pth")
torch.save(model,'output/full_model.pth')

# Plot accuracy
plt.plot(accuracy_list)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.savefig('output/model_accuracy.png')

t1_stop = process_time()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 



