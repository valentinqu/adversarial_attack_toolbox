import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import ToTensor, Normalize
import numpy as np
from PIL import Image
import csv
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

    print("Example image shape:", x_train[0].shape)
    
    return x_train, y_train, x_test, y_test, min_value, max_value

def load_mydata_nlp():
    """
    Load local NLP dataset from mydata/nlp/train.csv and mydata/nlp/test.csv.
    CSVs must contain columns: 'text' and 'label'.
    Returns lists of texts and labels for train/test, plus dummy min/max.
    """
    # New directory-based paths (preferred)
    train_dir_path = 'mydata/nlp/train/train.csv'
    test_dir_path = 'mydata/nlp/test/test.csv'
    # Backward-compatible single-file paths
    train_path = 'mydata/nlp/train.csv'
    test_path = 'mydata/nlp/test.csv'

    # Resolve actual train csv path
    if os.path.exists(train_dir_path):
        resolved_train = train_dir_path
    elif os.path.exists(train_path):
        resolved_train = train_path
    else:
        raise FileNotFoundError("NLP train.csv not found. Expected at mydata/nlp/train/train.csv or mydata/nlp/train.csv")

    df_train = pd.read_csv(resolved_train)
    if 'text' not in df_train.columns or 'label' not in df_train.columns:
        raise ValueError("train.csv must contain 'text' and 'label' columns")

    x_train = df_train['text'].astype(str).tolist()
    y_train = df_train['label'].tolist()

    # Resolve test csv if exists
    if os.path.exists(test_dir_path):
        resolved_test = test_dir_path
    elif os.path.exists(test_path):
        resolved_test = test_path
    else:
        resolved_test = None

    if resolved_test is not None:
        df_test = pd.read_csv(resolved_test)
        if 'text' not in df_test.columns or 'label' not in df_test.columns:
            raise ValueError("test.csv must contain 'text' and 'label' columns")
        x_test = df_test['text'].astype(str).tolist()
        y_test = df_test['label'].tolist()
    else:
        # If no test.csv, split from train
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, stratify=y_train if len(set(y_train)) > 1 else None
        )

    # NLP does not use min/max in current pipelines; return dummies
    min_value, max_value = 0.0, 1.0

    return x_train, y_train, x_test, y_test, min_value, max_value

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, min_value, max_value = load_mydata()
    
