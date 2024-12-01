import torchvision.transforms.functional as trF
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# CONSTANTS
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
NORMALIZED_MEAN = [0.485, 0.456, 0.406]
NORMALIZED_STD = [0.229, 0.224, 0.225]

# get current directory
dataset_path = "/scratch/hfontaine/datasets/PreprocessedWider"
work_path = "/home/hfontaine/MAX78000_face_detection/"

# Custom dataset class to read preprocessed images
class PreprocessedWiderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpeg')]
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image and label
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)  # Open image
        image = trF.to_tensor(image)  # Convert image to tensor
        image = trF.normalize(image, NORMALIZED_MEAN, NORMALIZED_STD)  # Normalize image
        # Extract the label from the filename (before the '.jpeg' extension)
        label = int(image_name.split('_')[-1][0])  # 'image_00_1.jpeg' -> label = 1
        if label :
            return image, torch.tensor([1, 0], dtype=torch.float32) 
        else:
            return image, torch.tensor([0, 1], dtype=torch.float32)


# Example of how to load the preprocessed dataset
dataset_train = PreprocessedWiderDataset(root_dir=os.path.join(dataset_path, 'train'))
dataset_val = PreprocessedWiderDataset(root_dir=os.path.join(dataset_path, 'val'))

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=1) # sugested num_workers=1
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=1) # sugested num_workers=1

# print some information about the dataset
print("Number of training images: ", len(dataset_train))

import torch.nn as nn
import torchsummary

class FaceDetector(nn.Module) :
    def __init__(self, num_channels=3, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH) :
        super(FaceDetector, self).__init__()
        self.h_dim = image_height
        self.w_dim = image_width
    
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # padding = 1 -> no change in size
        # maxpooling 2x2 -> both dimensions divided by 4
        self.h_dim //= 2  # 32
        self.w_dim //= 2  # 32

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2)
        # padding = 1 -> no change in size
        # maxpooling 2x2 -> both dimensions divided by 4
        self.h_dim //= 2  # 16
        self.w_dim //= 2  # 16 

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2)
        # padding = 1 -> no change in size
        # maxpooling 3x4 -> height divided by 3, width divided by 4
        self.h_dim //= 2  # 8
        self.w_dim //= 2  # 8

        self.linear = nn.Linear(self.h_dim * self.w_dim * 64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(-1, self.h_dim * self.w_dim * 64)
        x = self.linear(x)
        x = self.softmax(x)
        return x


# Initialize model and move it to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = FaceDetector().to(device)

# Print the model architecture
torchsummary.summary(face_detector, (3, IMAGE_HEIGHT, IMAGE_WIDTH), device='cuda')

from tqdm import tqdm
import torch.optim as optim

# Define optimizer
optimizer = optim.Adam(face_detector.parameters(), lr=0.001)

# Define training function with progress bar
def train(model, loader, optimizer, loss_fn, epoch, num_epochs):
    model.train()
    total_loss = 0
    # Add a tqdm progress bar for training batches
    with tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False) as pbar:
        for i, (images, targets) in enumerate(pbar):
            images = images.to(device)  # Move data to GPU
            targets = targets.to(device)  # Move targets to GPU
            
            optimizer.zero_grad() # Zero the gradients
            outputs = model(images) # Forward pass
            loss = loss_fn(outputs, targets) # Compute the loss
            loss.backward()  #| Backward pass and optimization
            optimizer.step() #|

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (i + 1)) # Update the progress bar
    return total_loss / len(loader)

# Define validation function
def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    train_loss = train(face_detector, train_loader, optimizer, loss_fn, epoch, num_epochs)
    val_loss = validate(face_detector, val_loader, loss_fn)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the model
torch.save(face_detector.state_dict(), os.path.join(work_path, 'face_detector.pth'))