import torch
import torch.nn as nn
import torch.nn.functional as F
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

#from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset
from art import metrics
from art.estimators.classification import PyTorchClassifier
from art.metrics.privacy.membership_leakage import SHAPr
from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from art.utils import to_categorical

from lime import lime_image
# from skimage.segmentation import mark_boundaries



class CleverScoreCalculator(PyTorchClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        nb_classes: int,  # 这里应该是一个整数
        input_shape: Tuple[int, ...],
        loss: nn.Module = nn.CrossEntropyLoss(), 
        optimizer: torch.optim.Optimizer = None,
        device_type: str = "gpu",
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: str = "dynamic",
        channels_first: bool = True,
        clip_values: tuple = (0.0, 1.0),
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing: tuple = (0.0, 1.0),
    ):
        """
        CleverScoreCalculator 初始化，继承 PyTorchClassifier 并添加 CLEVER 分数计算功能。

        :param model: PyTorch 模型实例。
        :param loss: 损失函数。
        :param input_shape: 输入数据的形状。
        :param nb_classes: 分类的类别数量。
        :param optimizer: 优化器实例。
        :param use_amp: 是否使用自动混合精度训练。
        :param opt_level: 混合精度的优化级别。
        :param loss_scale: 混合精度训练时的损失缩放。
        :param channels_first: 是否将通道放在第一个维度。
        :param clip_values: 输入数据的剪切值范围。
        :param preprocessing_defences: 数据预处理防御方法。
        :param postprocessing_defences: 数据后处理防御方法。
        :param preprocessing: 预处理数据时使用的减法项和除法项。
        :param device_type: 运行模型的设备类型，可以是 'cpu' 或 'gpu'。
        """
        # 调用父类 PyTorchClassifier 的初始化方法
        super().__init__(
            model=model,
            nb_classes=nb_classes,  # 在这里传递正确的类别数量
            input_shape=input_shape,
            loss=loss,
            optimizer=optimizer,
            device_type=device_type,
            use_amp=use_amp,
            opt_level=opt_level,
            loss_scale=loss_scale,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            
        )

        # 设置设备类型
        #self.device_type = device_type
        #self._device = torch.device("cuda" if torch.cuda.is_available() and device_type == 'gpu' else "cpu")
        #self.model = model.to(self._device)  # 将模型移动到指定设备
        

    def compute_untargeted_clever(self, x_sample, nb_batches=50, batch_size=10, radius=5, norm=1):
        """Compute the untargeted CLEVER score."""
        clever_untargeted = metrics.clever_u(self, x_sample, nb_batches, batch_size, radius, norm)
        return clever_untargeted

    def compute_targeted_clever(self, x_sample, target_class, nb_batches=50, batch_size=10, radius=5, norm=1):
        """Compute the targeted CLEVER score."""
        clever_targeted = metrics.clever_t(
            classifier=self,
            x=x_sample,
            target_class=target_class,
            nb_batches=nb_batches,
            batch_size=batch_size,
            radius=radius,
            norm=norm
        )
        return clever_targeted
    
    def computer_SHAPr(self, x_train, y_train, x_test, y_test):
        SHAPr_leakage = SHAPr(self, x_train, y_train, x_test, y_test)
        return SHAPr_leakage
    
    def predict_class(self, x_sample):
        """
        Get the predicted class from the model.

        :param x_sample: Sample to predict
        :return: Predicted class of the sample
        """
        predicted = self.predict(x_sample)
        return np.argmax(predicted)

    def _predict(self, images):
        """
        Prediction function for LIME.

        :param images: Input image data in the format (batch_size, height, width, channels).
        :return: Model's prediction output.
        """
        '''
        self.model.eval()

        batch = torch.stack(tuple(i for i in images), dim=0)

        # Put data on the GPU
        batch = batch.to(self._device)
        # Calculate classification result
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
        '''
        images = torch.tensor(images).float().to(self._device)

        images = images.permute(0, 3, 1, 2)
        with torch.no_grad():
            outputs = self.model(images)
        return outputs.cpu().numpy()

    def explain_with_lime(self, sample, top_labels=1, hide_color=1, num_samples=1000):
        """
        Generate explanations for an image using LIME.

        :param sample: A single image sample.
        :param top_labels: Number of top labels to explain, default is 1.
        :param hide_color: Color to hide, default is 1.
        :param num_samples: Number of samples for LIME, default is 1000.
        :return: Explanation object generated by LIME.
        """
        # Create a LIME image explainer
        explainer = lime_image.LimeImageExplainer()

        # Generate explanation
        explanation = explainer.explain_instance(
            sample,
            self._predict,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples
        )

        return explanation

def load_mnist_dataset():
    # 定义数据集转换（数据标准化等）
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))  # 标准化
        
    ])

    # 加载 CIFAR-10 训练集和测试集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 提取整个训练集和测试集
    x_train, y_train = torch.utils.data.default_collate(trainset)
    x_test, y_test = torch.utils.data.default_collate(testset)

    min_value = x_train.min()
    max_value = x_train.max()

    return x_train, y_train, x_test, y_test, min_value, max_value

def load_cifar10_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
         ])


    # 加载 CIFAR-10 训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 提取整个训练集和测试集
    x_train, y_train = torch.utils.data.default_collate(trainset)
    x_test, y_test = torch.utils.data.default_collate(testset)

    min_value = x_train.min()
    max_value = x_train.max()

    return x_train, y_train, x_test, y_test, min_value, max_value

# for task posion
def select_trigger_train(x_train, y_train, K, class_source, class_target):
    x_train_ = np.copy(x_train)
    
    # 在一维标签数组中直接找到匹配 class_source 和 class_target 的索引
    index_source = np.where(y_train == class_source)[0][:K]
    index_target = np.where(y_train == class_target)[0]
    
    x_trigger = x_train_[index_source]
    y_trigger = to_categorical([class_target], nb_classes=10)
    y_trigger = np.tile(y_trigger, (len(index_source), 1))
    
    return x_trigger, y_trigger, index_target

def save_poisoned_data(x_poison, y_poison, class_source, class_target):
    # 创建文件名，包含目标类别和来源类别
    file_name_x = f"x_poison_source_{class_source}_target_{class_target}.npy"
    file_name_y = f"y_poison_source_{class_source}_target_{class_target}.npy"

    # 保存数据到本地
    np.save(file_name_x, x_poison)
    np.save(file_name_y, y_poison)

    print(f"Poisoned data saved as {file_name_x} and {file_name_y}")

def add_trigger_patch(x_set, trigger_path, patch_size, patch_type="fixed"):
    print(x_set.shape)
    img = Image.open(trigger_path)
    numpydata = asarray(img)
    print("shape of numpydata",numpydata.shape)
    patch = resize(numpydata, (patch_size, patch_size, 3))
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
    return np.eye(nb_classes)[y.numpy()] 
# for LIME explain
class ToRGBTransform:
    def __call__(self, img):
        # 将灰度图像转换为伪 RGB 格式
        img_rgb = img.repeat(3, 1, 1)  # 复制单通道到三个通道
        return img_rgb

def get_transform_for_channels(num_channels):
    if num_channels == 1:
        # 对单通道图像（例如灰度图像）应用 ToRGBTransform
        return transforms.Compose([
            transforms.ToTensor(),
            ToRGBTransform()
        ])
    else:
        # 对 RGB 图像应用常规变换
        return transforms.Compose([
            transforms.ToTensor()
            # 根据需要添加其他变换
        ])
    
if __name__ == "__main__":
    pass
