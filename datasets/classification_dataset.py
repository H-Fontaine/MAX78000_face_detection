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
                if path.endswith('.jpg') :
                    name = path.split('.')[0]
                    image_path = os.path.join(class_path, f"{name}.jpg")
                    self.images.append((image_path, c))
                    bbox_path = os.path.join(class_path, f"{name}.txt")
                    self.bboxes.append(bbox_path)
    
    def __getitem__(self, index):
        image_path, is_face = self.images[index]
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
def get_datasets(data, load_train=False, load_test=False, fold_ratio=1):
   
    (data_dir, args) = data
    #data_dir = data

    transform = transforms.Compose([
        ai8x.normalize(args),
        ai8x.fold(fold_ratio=fold_ratio)
    ])

    if load_train:
        train_dataset = ClassificationDataset(root=data_dir, split='train', transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = ClassificationDataset(root=data_dir, split='val', transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def get_untouched(data, load_train=False, load_test=False):
    return get_datasets(data, load_train, load_test, 1)

def get_folded(data, load_train=False, load_test=False):
    return get_datasets(data, load_train, load_test, 4)

"""
Dataset description
"""
datasets = [
    {
        'name': 'classification',
        'input': (3, 88, 88),
        'output': list(map(str, range(2))),
        'loader': get_untouched,
    },
    {
        'name': 'classification_folded',
        'input': (48, 22, 22),
        'output': list(map(str, range(2))),
        'loader': get_folded,
    }
]

class ARGS :
    act_mode_8bit = False


if __name__ == '__main__':
    dataset_train, dataset_test = get_datasets(("datasets/classification", ARGS()), True, True)
    dataloader = DataLoader(dataset_train, batch_size=4,
                        shuffle=True, num_workers=0)

    fig, ax = plt.subplots(4, 4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(),
            sample_batched[1].size())

        # observe 4th batch and stop.
        if i_batch < 4:
            for i, img in enumerate(sample_batched[0]):
                print(img.shape)
                ax[i_batch, i].imshow(img.permute((1,2,0)))
        else:
            break
                
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.ioff()
    plt.show()