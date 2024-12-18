from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

from torchvision import transforms
from PIL import Image

import ai8x

import os

import matplotlib.pyplot as plt

"""
Custom image dataset class
"""
class ClassificationDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.classes = [d.name for d in os.scandir(f"{root}/{split}") if d.is_dir()]
        self.images = []
        self.bboxes = []
        for c in self.classes :
            class_path = os.path.join(f"{root}/{split}", c)
            for path in os.listdir(class_path) :
                if path.endswith('.png') :
                    name = path.split('.')[0]
                    image_path = os.path.join(class_path, f"{name}.png")
                    self.images.append((image_path, c))
                    bbox_path = os.path.join(class_path, f"{name}.txt")
                    self.bboxes.append(bbox_path)
    
    def __getitem__(self, index):
        image_path, classe = self.images[index]
        is_face = 1 if classe == 'face' else 0
        image = Image.open(image_path)
        image = transforms.ToTensor()(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(is_face)
        
    def __len__(self):
        return len(self.images)

"""
Dataloader function
"""
def get_datasets(data, load_train=False, load_test=False):
   
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(),
        ai8x.normalize(args)
    ])

    if load_train:
        train_dataset = ClassificationDataset(root=data_dir, split='train', transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = ClassificationDataset(root=data_dir, split='test', transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'homemade',
        'input': (3, 88, 88),
        'output': list(map(str, range(2))),
        'loader': get_datasets,
    }
]

class ARGS :
    act_mode_8bit = False


if __name__ == '__main__':
    dataset_train, dataset_test = get_datasets(("datasets/homemade", ARGS()), True, True)
    dataloader = DataLoader(dataset_train, batch_size=4,
                        shuffle=True, num_workers=0)

    fig, ax = plt.subplots(4, 4)

    for i_batch, sample_batched in enumerate(dataloader):
        # observe 4th batch and stop.
        if i_batch < 4:
            for i, img in enumerate(sample_batched[0]):
                ax[i_batch, i].imshow(img.permute((1,2,0)))
        else:
            break
                
    plt.axis('off')
    plt.ioff()
    plt.show()