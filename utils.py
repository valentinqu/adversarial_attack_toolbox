import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
#import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm import trange
import numpy as np
import os, sys
import pdb
from PIL import Image
from numpy import asarray
from skimage.transform import resize
import random

from art import metrics
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel

#from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from datasets import load_dataset
from art import metrics
from art.estimators.classification import PyTorchClassifier
from art.metrics.privacy.membership_leakage import SHAPr
from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from art.utils import to_categorical

import hnswlib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

from lime import lime_image
# from skimage.segmentation import mark_boundaries

def load_mnist_dataset():
    # Define dataset transformations (data normalization, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 training and testing sets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Extract the entire training and test sets
    x_train, y_train = torch.utils.data.default_collate(trainset)
    x_test, y_test = torch.utils.data.default_collate(testset)

    min_value = x_train.min()
    max_value = x_train.max()

    return x_train, y_train, x_test, y_test, min_value, max_value

def load_cifar10_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Load CIFAR-10 training and testing sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Extract the entire training and test sets
    x_train, y_train = torch.utils.data.default_collate(trainset)
    x_test, y_test = torch.utils.data.default_collate(testset)

    min_value = x_train.min()
    max_value = x_train.max()

    return x_train, y_train, x_test, y_test, min_value, max_value

def load_imdb_dataset():
    """
    Load the IMDb dataset for sentiment analysis.

    Returns:
        x_train (list): List of text reviews from the training set.
        y_train (list): List of sentiment labels (0 or 1) from the training set.
        x_test (list): List of text reviews from the test set.
        y_test (list): List of sentiment labels (0 or 1) from the test set.
        min_value (None): Placeholder for consistency with image datasets.
        max_value (None): Placeholder for consistency with image datasets.
    """
    # Load the IMDb dataset
    dataset = load_dataset('imdb')
    train_data = dataset['train']
    test_data = dataset['test']

    # Extract text reviews and labels
    x_train = train_data['text']   # List of strings (reviews)
    y_train = train_data['label']  # List of integers (0 or 1)

    x_test = test_data['text']
    y_test = test_data['label']

    # Since text data has no numerical min/max values, return None
    min_value = None
    max_value = None

    return x_train, y_train, x_test, y_test, min_value, max_value

def load_hf_dataset(dataset_name, text_field='text', label_field='label'):
    """
    Load a dataset from the Hugging Face datasets library.

    Args:
        dataset_name (str): Name of the dataset to load.
        text_field (str): Text field name, default is 'text'.
        label_field (str): Label field name, default is 'label'.

    returns:
        tuple: x_train, y_train, x_test, y_test, min_value, max_value
    """
    try:
        
        dataset = load_dataset(dataset_name)

        # check if text and label fields exist in the dataset
        if text_field not in dataset['train'].features:
            raise ValueError(f"Text field '{text_field}' not found in dataset.")
        if label_field not in dataset['train'].features:
            raise ValueError(f"Label field '{label_field}' not found in dataset.")

       
        train_data = dataset['train']
        test_data = dataset['test']

        x_train = train_data[text_field]  
        y_train = train_data[label_field]  

        x_test = test_data[text_field]
        y_test = test_data[label_field]

        # Since text data has no numerical min/max values, return None  
        min_value = None
        max_value = None

        print(f"Dataset '{dataset_name}' loaded successfully.")
        print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

        return x_train, y_train, x_test, y_test, min_value, max_value

    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {str(e)}")
        return None, None, None, None, None, None
    

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze()       # shape: [max_length]
        attention_mask = encoding['attention_mask'].squeeze()  # shape: [max_length]

        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)
        else:
            return input_ids, attention_mask

class BertClassifier(nn.Module):
    """
    Wrap BertForSequenceClassification for ART's PyTorchClassifier.
    The forward input x should contain concatenated input_ids and attention_mask.
    """
    def __init__(self, model, max_length=128):
        super(BertClassifier, self).__init__()
        self.model = model
        self.max_length = max_length

    def forward(self, x):
        # x shape: [batch_size, max_length * 2]
        input_ids = x[:, :self.max_length].long()
        attention_mask = x[:, self.max_length:].long()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits
# For task poisoning
def select_trigger_train(x_train, y_train, K, class_source, class_target):
    x_train_ = np.copy(x_train)
    
    # Directly find the indices that match class_source and class_target in the one-dimensional label array
    index_source = np.where(y_train == class_source)[0][:K]
    index_target = np.where(y_train == class_target)[0]
    
    x_trigger = x_train_[index_source]
    y_trigger = to_categorical([class_target], nb_classes=10)
    y_trigger = np.tile(y_trigger, (len(index_source), 1))
    
    return x_trigger, y_trigger, index_target

def save_poisoned_data(x_poison, y_poison, class_source, class_target):
    # Create file names including target and source class
    file_name_x = f"x_poison_source_{class_source}_target_{class_target}.npy"
    file_name_y = f"y_poison_source_{class_source}_target_{class_target}.npy"

    # Save data locally
    np.save(file_name_x, x_poison)
    np.save(file_name_y, y_poison)

    print(f"Poisoned data saved as {file_name_x} and {file_name_y}")

def add_trigger_patch(x_set, trigger_path, patch_size, input_channel, patch_type="fixed"):
    #print(x_set.shape)
    img = Image.open(trigger_path)
    numpydata = asarray(img)
    #print("shape of numpydata",numpydata.shape)
    patch = resize(numpydata, (patch_size, patch_size, input_channel))
    patch = np.transpose(patch, (2, 0, 1))
    print("shape of patch", patch.shape)
    if patch_type == "fixed":
        x_set[:, :, -patch_size:, -patch_size:] = patch
    else:
        for x in x_set:
            x_cord = random.randrange(0,x.shape[1] - patch.shape[1] + 1)
            y_cord = random.randrange(0,x.shape[2] - patch.shape[2] + 1)
            x[:,x_cord:x_cord+patch_size,y_cord:y_cord+patch_size]=patch

    return x_set

def to_one_hot(y, nb_classes):
    return np.eye(nb_classes)[y] 

def hnsw(features, k=10, ef=100, M=48):
    """使用 HNSW 构建最近邻图"""
    num_samples, dim = features.shape
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)
    neighs, weight = p.knn_query(features, k + 1)
    return neighs, weight

def construct_adj(neighs, weight):
    """构建邻接矩阵"""
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1
    idx0 = np.arange(dim)
    row = np.repeat(idx0.reshape(-1,1), k, axis=1).reshape(-1,)
    col = neighs[:, 1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    return adj

def spade_score(input_features, output_features, k=10):
    """计算 SPADE 分数"""
    # 构建输入和输出的 k-NN 图
    neighs_in, dist_in = hnsw(input_features, k)
    adj_in = construct_adj(neighs_in, dist_in)
    neighs_out, dist_out = hnsw(output_features, k)
    adj_out = construct_adj(neighs_out, dist_out)

    # 拉普拉斯矩阵
    L_in = laplacian(adj_in, normed=True)
    L_out = laplacian(adj_out, normed=True)

    # 特征值分解
    eigvals_in, eigvecs_in = np.linalg.eig(L_in.toarray())
    eigvals_out, eigvecs_out = np.linalg.eig(L_out.toarray())

    spade_score_value = max(eigvals_out) / max(eigvals_in)
    return spade_score_value

if __name__ == "__main__":
    pass