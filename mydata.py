import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import ToTensor, Normalize
import numpy as np
from PIL import Image
class myImageDataset(Dataset):
    def __init__(self, img_dir, img_label_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(img_label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def compute_min_max(dataset):
    all_images = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        all_images.append(image.numpy())
    all_images = np.concatenate(all_images, axis=0)
    min_value = np.min(all_images)
    max_value = np.max(all_images)
    return min_value, max_value

def split_dataset(dataset, test_size=0.2):
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    
    return train_set, test_set

def load_mydata():
    label_path = 'mydata/labels.csv'
    img_root_path = 'mydata/images/'
    
    dataset = myImageDataset(img_root_path, label_path, transform=ToTensor())
    
    min_value, max_value = compute_min_max(dataset)
    print(f"Min pixel value: {min_value}")
    print(f"Max pixel value: {max_value}")
    
    train_set, test_set = split_dataset(dataset)
    
    x_train = [dataset[i][0] for i in range(len(train_set))]
    y_train = [dataset[i][1] for i in range(len(train_set))]
    x_test = [dataset[i][0] for i in range(len(test_set))]
    y_test = [dataset[i][1] for i in range(len(test_set))]
    
    return x_train, y_train, x_test, y_test, min_value, max_value

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, min_value, max_value = load_mydata()
    
