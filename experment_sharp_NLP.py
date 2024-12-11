import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from art.estimators.classification import PyTorchClassifier
from art.metrics import SHAPr
from datasets import load_dataset

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
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
        # 将 input_ids 和 attention_mask 拼接起来
        combined = torch.cat((encoding['input_ids'], encoding['attention_mask']), dim=1).squeeze()
        return {
            'input': combined,  # [max_length * 2]
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(texts, labels, tokenizer, max_length, batch_size):
    ds = IMDbDataset(texts, labels, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size)

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载预训练的 BERT 模型
    print("Loading pre-trained BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.eval()
    model.to(device)

    # 加载 IMDb 数据集
    print("Loading IMDb dataset...")
    dataset = load_dataset('imdb')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # 使用预训练的 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 创建数据加载器
    max_length = 128
    batch_size = 32

    subset = False
    if subset:
        # 为了节省计算资源，可以使用部分数据
        train_texts = train_texts[:1000]
        train_labels = train_labels[:1000]
        test_texts = test_texts[:1000]
        test_labels = test_labels[:1000]
    else:
        train_texts = train_texts
        train_labels = train_labels
        test_texts = test_texts
        test_labels = test_labels

    train_loader = create_data_loader(train_texts, train_labels, tokenizer, max_length, batch_size)
    test_loader = create_data_loader(test_texts, test_labels, tokenizer, max_length, batch_size)

    # 准备输入和标签
    def get_features_and_labels(data_loader):
        inputs_list = []
        labels_list = []
        for batch in data_loader:
            inputs_list.append(batch['input'])
            labels_list.append(batch['labels'])
        inputs = torch.cat(inputs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return inputs, labels

    print("Preparing data...")
    x_train_inputs, y_train = get_features_and_labels(train_loader)
    x_test_inputs, y_test = get_features_and_labels(test_loader)

    # 将数据移动到设备
    x_train_inputs = x_train_inputs.to(device)
    y_train = y_train.to(device)

    x_test_inputs = x_test_inputs.to(device)
    y_test = y_test.to(device)

    # 定义一个包装模型，用于 ART 的 PyTorchClassifier
    class BertClassifier(nn.Module):
        def __init__(self, model, max_length):
            super(BertClassifier, self).__init__()
            self.model = model
            self.max_length = max_length

        def forward(self, x):
            # x 的形状为 [batch_size, max_length * 2]
            input_ids = x[:, :self.max_length].long()
            attention_mask = x[:, self.max_length:].long()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            return logits

    wrapped_model = BertClassifier(model, max_length)
    wrapped_model.eval()
    wrapped_model.to(device)

    # 创建 ART 的 PyTorchClassifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(wrapped_model.parameters(), lr=2e-5)

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(max_length * 2,),
        nb_classes=2,
        device_type='gpu' if torch.cuda.is_available() else 'cpu',
    )

    # 将数据从 GPU 移动到 CPU，并转换为 NumPy 数组
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    x_train_np = to_numpy(x_train_inputs)
    x_test_np = to_numpy(x_test_inputs)
    y_train_np = to_numpy(y_train)
    y_test_np = to_numpy(y_test)

    print("Calculating SHAPr leakage...")
    SHAPr_leakage = SHAPr(
        classifier,
        x_train_np,
        y_train_np,
        x_test_np,
        y_test_np,
        enable_logging=True
    )
    print("Average SHAPr leakage: ", np.average(SHAPr_leakage))

    