import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset
from art.metrics import SHAPr

if __name__ == "__main__":

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载预训练的模型
    print(f'Loading pre-trained model...')
    # 从 torch.hub 加载在 CIFAR-10 上预训练的 ResNet-20 模型
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
    model.eval()
    model = model.to(device)

    # 加载 CIFAR-10 数据集
    print(f'Loading dataset...')    
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset('cifar10')
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    # 将标签从 one-hot 编码转换为类别索引
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # 定义损失函数和优化器（虽然不需要训练，但 PyTorchClassifier 需要这些参数）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 创建 ART 分类器
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    # 打印输入形状
    print(x_train.shape)
    print(f"Expected input shape: {classifier.input_shape}")

    # 计算 SHAPr 泄漏
    print("Calculating SHAPr leakage...")

    # 将数据转换为 PyTorch 张量，并移动到 GPU
    x_train_tensor = torch.tensor(x_train).to(device)
    y_train_tensor = torch.tensor(y_train).to(device)
    x_test_tensor = torch.tensor(x_test).to(device)
    y_test_tensor = torch.tensor(y_test).to(device)

    # **在调用 SHAPr 之前，将数据和标签移动到 CPU 并转换为 NumPy 数组**
    x_train_np = x_train_tensor.cpu().numpy()
    y_train_np = y_train_tensor.cpu().numpy()
    x_test_np = x_test_tensor.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    # 调用 SHAPr 函数
    SHAPr_leakage = SHAPr(
        classifier, 
        x_train_np, 
        y_train_np, 
        x_test_np, 
        y_test_np, 
        enable_logging=True
    )
    print("Average SHAPr leakage: ", np.average(SHAPr_leakage))
