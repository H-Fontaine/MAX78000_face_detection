from torchvision.datasets import WIDERFace
from torchvision import transforms
import torch.nn.functional as nnF
import torchvision.transforms.functional as trF
import torch
import random as rd


#CONSTANTS
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NORMALIZED_MEAN = [0.485, 0.456, 0.406]
NORMALIZED_STD = [0.229, 0.224, 0.225]

#get current directory
dataset_path = "/scratch/hfontaine/datasets"

class WiderDatasetBBOX(WIDERFace):
    def __init__(self, root, split, height, width, transform=None, target_transform=None, download=True):
        super(WiderDatasetBBOX, self).__init__(root, split, None, None, download)
        self.height = height
        self.width = width
        if(transform != None or target_transform != None):
            print("Warning: transform and target_transform are not supported and will be ignored")
    
    def __getitem__(self, index):
        image, target = super(WiderDatasetBBOX, self).__getitem__(index)
        image = transforms.ToTensor()(image)
        im_height, im_width = image.shape[1:]

        #pad image if smaller than required size
        padl = padr = padt = padb = 0
        if im_width < self.width :
            padl = (self.width - im_width) // 2
            padr = self.width - im_width - padl
            image = nnF.pad(image, (padl, padr), mode='constant', value=0)
        if im_height < self.height :
            padt = (self.height - im_height) // 2
            padb = self.height - im_height - padt
            image = nnF.pad(image, (0, 0, padt, padb), mode='constant', value=0)
        im_height, im_width = image.shape[1:]

        #crop image at random location
        crop_x = rd.randint(0, im_width - self.width)
        crop_y = rd.randint(0, im_height - self.height)
        image = trF.crop(image, crop_y, crop_x, self.height, self.width)

        #normalize image
        image = transforms.Normalize(mean=NORMALIZED_MEAN, std=NORMALIZED_STD)(image)
        
        #transform bounding boxes
        bbox = []
        for i in range(len(target['bbox'])) :
            x, y, w, h = target['bbox'][i]
            x = int(x) + padl; y = int(y) + padl; w = int(w); h = int(h)
            if x >= crop_x + self.width or y >= crop_y + self.height :
                continue
            if x + w <= crop_x or y + h <= crop_y :
                continue
            x1 = max(crop_x, x); w_new = w-(x1-x)
            y1 = max(crop_y, y); h_new = h-(y1-y)
            if x1 + w_new > crop_x + self.width :
                w_new = crop_x + self.width - x1
            if y1 + h_new > crop_y + self.height :
                h_new = crop_y + self.height - y1
            area_new = w_new*h_new
            if area_new >= 0.4*w*h :
                bbox.append([x1-crop_x, y1-crop_y, w_new, h_new])
                
        bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox.view(-1, 4)
        return image, bbox



class WiderDatasetSimple(WiderDatasetBBOX) :
    def __init__(self, root, split, height, width, transform=None, target_transform=None, download=True) :
        super(WiderDatasetSimple, self).__init__(root, split, height, width, transform, target_transform, download)

    def __getitem__(self, index):
        image, target = super(WiderDatasetSimple, self).__getitem__(index)
        if len(target) == 0 :
            return image, torch.tensor([1, 0], dtype=torch.float32) #no face
        else :
            return image, torch.tensor([0, 1], dtype=torch.float32) #face

#download WIDER_face dataset with pythorch
dataset_train_bbox = WiderDatasetBBOX(root=dataset_path, split='train', height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
dataset_val_bbox = WiderDatasetBBOX(root=dataset_path, split='train', height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
dataset_test_bbox = WiderDatasetBBOX(root=dataset_path, split='train', height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

dataset_train_simple = WiderDatasetSimple(root=dataset_path, split='train', height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
dataset_val_simple = WiderDatasetSimple(root=dataset_path, split='train', height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
dataset_test_simple = WiderDatasetSimple(root=dataset_path, split='train', height=IMAGE_HEIGHT, width=IMAGE_WIDTH)


#print some information about the dataset
print("Number of training images: ", len(dataset_train_bbox))
print("Number of validation images: ", len(dataset_val_bbox))
print("Number of test images: ", len(dataset_test_bbox))



from torch.utils.data import DataLoader

batch_size = 128
load_train = DataLoader(dataset_train_simple, batch_size=batch_size)
load_val = DataLoader(dataset_val_simple, batch_size=batch_size)
load_test = DataLoader(dataset_test_simple, batch_size=batch_size)





import torchvision.models as models
import torch.nn as nn
import torchsummary

class FaceDetector(nn.Module) :
    def __init__(self, num_channels = 3, image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH) :
        super(FaceDetector, self).__init__()
        self.h_dim = image_height
        self.w_dim = image_width
    
        self.conv1 = nn.Conv2d(num_channels, 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((4, 4), (4, 4))
        #padding = 1 -> no change in size
        #maxpooling 4x4 -> both dimensions divided by 4
        self.h_dim //= 4 #120
        self.w_dim //= 4 #160

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((4, 4), (4, 4))
        #padding = 1 -> no change in size
        #maxpooling 4x4 -> both dimensions divided by 4
        self.h_dim //= 4 #30
        self.w_dim //= 4 #40 

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((3, 4), (3, 4))
        #padding = 1 -> no change in size
        #maxpooling 3x4 -> height divided by 3, width divided by 4
        self.h_dim //= 3 #10
        self.w_dim //= 4 #10

        self.linear = nn.Linear(self.h_dim * self.w_dim * 16, 2)
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
        x = x.view(-1, self.h_dim * self.w_dim * 16)
        x = self.linear(x)
        x = self.softmax(x)
        return x
    
face_detector = FaceDetector()
face_detector.cuda()

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
            images = images
            targets = targets
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update progress bar with current batch loss
            pbar.set_postfix(batch_loss=loss.item())
    return total_loss / len(loader)

# Define validation function with progress bar
def validate(model, loader, loss_fn, epoch, num_epochs):
    model.eval()
    total_loss = 0

    # Add a tqdm progress bar for validation batches
    with tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False) as pbar:
        with torch.no_grad():
            for i, (images, targets) in enumerate(pbar):
                images = images
                targets = targets
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()

                # Update progress bar with current batch loss
                pbar.set_postfix(batch_loss=loss.item())
    return total_loss / len(loader)

# Define test function
def test(model, loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        with tqdm(loader, desc="Testing", leave=False) as pbar:
            for i, (images, targets) in enumerate(pbar):
                images = images
                targets = targets
                outputs = model(images)
                loss = nn.BCELoss()(outputs, targets)
                total_loss += loss.item()

                # Update progress bar with current batch loss
                pbar.set_postfix(batch_loss=loss.item())
        print(f"Test Loss: {total_loss / len(loader)}")

# Train model with epoch progress
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(face_detector, load_train, optimizer, nn.BCELoss(), epoch, num_epochs)
    val_loss = validate(face_detector, load_val, nn.BCELoss(), epoch, num_epochs)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Test model
test(face_detector, load_test)

# Save model
torch.save(face_detector.state_dict(), "/home/hfontaine/MAX78000_face_detection/face_detector.pth")