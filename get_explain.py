import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Net

import ruamel.yaml as yaml
import argparse

from art import metrics
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset
from art import metrics

from lime import lime_image
from skimage.segmentation import mark_boundaries
from utils import CleverScoreCalculator, load_cifar10

def main(input_shape, nb_classes):
    # 读取 YAML 配置文件
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open('config.yaml', 'r') as file:
        config = yaml_loader.load(file)
    # 获取模型路径
    load_path = config['model_path']
    
    # import data
    # 定义转换（如需要，可以添加更多的转换操作）
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # 加载图片文件夹
    dataset = datasets.ImageFolder(root='/Users/peilinyue/Documents/work/adversarial_attack_toolbox/images_upload', transform=transform)
    #(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset('mnist')
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    #load model
    model = Net()
    model.load_state_dict(torch.load(load_path))
    model.eval()

    # Create an instance of ‘CleverScoreCalculator’
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # Create an instance of ‘CleverScoreCalculator’
    clever_calculator = CleverScoreCalculator(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        optimizer=optimizer,
        device_type='cpu'
    )

    # Generate explanations using LIME
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        images = np.squeeze(images.numpy(), axis=0)
        sample = np.transpose(images, (1, 2, 0))  # Convert to (height, width, channels) format
        explanation = clever_calculator.explain_with_lime(sample)

        # Display the explanation results
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp, mask)
        #plt.imshow(img_boundry2)
        #plt.show()

        # 找到数据的最小值和最大值
        min_val = np.min(img_boundry2)
        max_val = np.max(img_boundry2)

        # 将数据归一化到 0-1 范围
        img_boundry2_normalized = (img_boundry2 - min_val) / (max_val - min_val)


        # Save the image to a file
        output_filename = f'explained_imags/explanation_{i}.png'  # File name for the saved image
        plt.imsave(output_filename, img_boundry2_normalized)
        print(f'Saved image: {output_filename}')


if __name__ == '__main__':

    #define a ArgumentParse Instance
    parser = argparse.ArgumentParser(
        prog='get_clever_scores',
        description='Calculate CLEVER scores',
    )
    parser.add_argument('-shape', '--input_shape', required=True, nargs='+')
    parser.add_argument('-class', '--nb_classes', required=True, type=int)
    args = parser.parse_args()

    main(tuple(args.input_shape), args.nb_classes)


    
