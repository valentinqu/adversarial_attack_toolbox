import argparse
import os
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage.segmentation import felzenszwalb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from art.attacks.poisoning import SleeperAgentAttack
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
from art.metrics.privacy.membership_leakage import SHAPr
from art.utils import to_categorical
from lime import lime_image
from GEEX import geex

from model import Net  # Ensure that the Net class is defined in your model.py
from mydata import load_mydata  # Ensure you have a file named mydata.py containing the load_mydata function
from utils_toolbox import *

def calculate_clever_scores(nb_classes, model, x_test, device_type='gpu'):
    """Calculate untargeted CLEVER scores."""
    device_type = '"cuda:0' if torch.cuda.is_available() else 'cpu'
    input_shape = x_test.shape[1:]
    clever_calculator = PyTorchClassifier(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        device_type=device_type,
    )

    scores = []
    for i in range(50):  # Use the first 50 test images
        image = x_test[i].numpy()  # Keep batch dimension
        clever_untargeted = clever_u(clever_calculator, image, nb_batches=50, batch_size=10, radius=5, norm=1)
        scores.append(clever_untargeted)

    clever_untargeted_avg = np.mean(scores)
    print('-----------FINAL RESULT-----------')
    print(f"Untargeted CLEVER score: {clever_untargeted_avg}")

def calculate_SHAPr(nb_classes, model, x_train, y_train, x_test, y_test, NLP, device_type='gpu'):
    """Calculate SHAPr leakage."""
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_length = 128

    if NLP:
        # Using TextDataset and DataLoader to transform text data into input_ids+attention_mask tensor
        train_set = TextDataset(x_train, y_train, max_length=max_length)
        test_set = TextDataset(x_test, y_test, max_length=max_length)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        def get_features_and_labels(data_loader):
            inputs_list = []
            labels_list = []
            for batch in data_loader:
                input_ids, attention_mask, labels = batch
                combined = torch.cat((input_ids, attention_mask), dim=1)  # [batch_size, max_length * 2]
                inputs_list.append(combined)
                labels_list.append(labels)
            inputs = torch.cat(inputs_list, dim=0)  # [N, max_length * 2]
            labels = torch.cat(labels_list, dim=0)  # [N]
            return inputs, labels

        print("Preparing IMDb data for SHAPr (NLP)...")
        x_train_tensor, y_train_tensor = get_features_and_labels(train_loader)
        x_test_tensor, y_test_tensor = get_features_and_labels(test_loader)

        # Tensor to Numpy array
        x_train_np = x_train_tensor.cpu().numpy()
        y_train_np = y_train_tensor.cpu().numpy()
        x_test_np = x_test_tensor.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

        # Encapsulating Models with BertClassifier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        classifier = PyTorchClassifier(
            model=BertClassifier(model, max_length),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(max_length * 2,),
            nb_classes=nb_classes,
            device_type='gpu' if torch.cuda.is_available() else 'cpu',
        )

        SHAPr_leakage = SHAPr(classifier, x_train_np, y_train_np, x_test_np, y_test_np, enable_logging=True)

    else:
        input_shape = x_train.shape[1:]
        classifier = PyTorchClassifier(
            model=model,
            input_shape=input_shape,
            nb_classes=nb_classes,
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.SGD(model.parameters(), lr=0.1),
            device_type=device_type,
        )
        SHAPr_leakage = SHAPr(classifier, x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy(), enable_logging=True)

    print("Average SHAPr leakage: ", np.mean(SHAPr_leakage))
    
def poison(nb_classes, test, trigger_path, patch_size, x_train, x_test, y_train, y_test, min_, max_, model):
    """Perform data poisoning attack."""
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_train_np = x_train.numpy().astype(np.float32)
    x_test_np = x_test.numpy().astype(np.float32)
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    y_train_np = to_one_hot(y_train_np, nb_classes)
    y_test_np = to_one_hot(y_test_np, nb_classes)

    min_, max_ = float(min_), float(max_)
    mean, std = x_train_np.mean(), x_train_np.std()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()
    
    model_art = PyTorchClassifier(
        model=model,
        input_shape=x_train_np.shape[1:],
        loss=loss_fn,
        optimizer=optimizer,
        nb_classes=nb_classes,
        clip_values=(min_, max_),
        preprocessing=(mean, std),
        device_type=device_type,
    )

    x_train_orig_np = np.copy(x_train_np)

    # test accuracy
    predictions_benign = model_art.predict(x_test_np)
    accuracy_benign = np.sum(np.argmax(predictions_benign, axis=1) == np.argmax(y_test_np, axis=1)) / len(y_test_np)
    print("Accuracy on benign test examples: {}%".format(accuracy_benign * 100))


    # Handle trigger patch
    img = Image.open(trigger_path)
    input_channel = x_train_np.shape[1:][0]

    patch = resize(np.asarray(img), (patch_size, patch_size, input_channel))  # Ensure the patch has the same dimensions as the data
    patch = np.transpose(patch, (2, 0, 1))

    class_source = 0
    class_target = 1
    K = 1000

    x_trigger, y_trigger, index_target = select_trigger_train(x_train_np, y_train_np.argmax(axis=1), K, class_source, class_target)
    
    attack = SleeperAgentAttack(
        classifier=model_art,
        percent_poison=0.50,
        max_trials=1,
        max_epochs=500,
        learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), [250, 350, 400, 430, 460]),
        epsilon=16/255,
        batch_size=500,
        verbose=1,
        indices_target=index_target,
        patching_strategy="random",
        selection_strategy="max-norm",
        patch=patch,
        retraining_factor=4,
        model_retrain=True,
        model_retraining_epoch=80,
        retrain_batch_size=128,
        class_source=class_source,
        class_target=class_target,
        device_name=device_type,
    )

    # Generate poisoned data
    x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train_np, y_train_np, x_test_np, y_test_np)
    save_poisoned_data(x_poison, y_poison, class_source, class_target)

    indices_poison = attack.get_poison_indices()
    print("number of indices_poison:", len(indices_poison))
    # np.save('indices_poison.npy', indices_poison)
    # print("indices_poison saved")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    model_poisoned = PyTorchClassifier(
        model=model,
        input_shape=x_train_np.shape[1:],
        loss=loss_fn,
        optimizer=optimizer,
        nb_classes=nb_classes,
        clip_values=(min_, max_),
        preprocessing=(mean, std),
        device_type=device_type,
    )

    # Check attack effect
    if test:
        model_poisoned.fit(x_poison, y_poison, batch_size=128, nb_epochs=150, verbose=1)
        # model_posioned.fit(x_poison, y_poison, batch_size=128, nb_epochs=1, verbose=1)
        predictions = model_poisoned.predict(x_test_np)
        accuracy_poisioned = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_np, axis=1)) / len(y_test_np)
        print("Accuracy on poisioned test examples: {}%".format(accuracy_poisioned * 100))

        index_source_train = np.where(y_train_np.argmax(axis=1) == class_source)[0]
        x_train_trigger = x_train_orig_np[index_source_train]
        x_train_trigger = add_trigger_patch(x_train_trigger, trigger_path, patch_size, input_channel, "random")
        result_poisoned_train = model_poisoned.predict(x_train_trigger)

        success_test = (np.argmax(result_poisoned_train, axis=1) == class_target).sum() / result_poisoned_train.shape[0]
        print(f'Test Success Rate(Probability of poisioned sample recognized as a target category{class_target} is {success_test}):')

def explain(nb_classes, num_channels, model):
    """Generate model explanations using LIME."""
    device_type = '"cuda:0' if torch.cuda.is_available() else 'cpu'

    from skimage.segmentation import felzenszwalb  # Ensure felzenszwalb is imported

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1) if num_channels == 1 else transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if num_channels == 1 else x),
    ])

    dataset = datasets.ImageFolder(root='images_upload', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_shape = (num_channels, dataset[0][0].shape[1], dataset[0][0].shape[2])

    classifier = PyTorchClassifier(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        device_type=device_type,
    )

    def predict_fn(images_np):
        # images_np shape: (N, H, W, C)
        # Convert to (N, C, H, W)
        images_tensor = torch.tensor(images_np).permute(0, 3, 1, 2).float()
        # Decide whether to convert the number of channels based on num_channels
        if num_channels == 1 and images_tensor.shape[1] != 1:
            # Convert image to single channel
            images_tensor = images_tensor[:, 0:1, :, :]
        outputs = classifier.predict(images_tensor)
        return outputs

    import os
    output_dir = 'explained_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (images, labels) in enumerate(dataloader):
        images = images.numpy()
        sample = images[0]  # Shape: (C, H, W)
        sample = np.transpose(sample, (1, 2, 0))  # Convert to (H, W, C)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            sample,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=1,
            num_samples=1000,
            # segmentation_fn=felzenszwalb,
            batch_size=1
        )

        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )
        img_boundary = mark_boundaries(temp, mask)

        output_filename = f'{output_dir}/explanation_{i}.png'
        plt.imsave(output_filename, img_boundary)
        print(f'Saved image: {output_filename}')

def explain_geex(nb_classes, num_channels, model):
    """Generate model explanations using GEEX, and save results to disk."""
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define transformations
    IMG_SIZE = (1, 28, 28)
    PIX_MEAN, PIX_STD = (0.1307, ), (0.3081, )

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(PIX_MEAN, PIX_STD)]
    )


    dataset = datasets.ImageFolder(root='images_upload', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_shape = (num_channels, dataset[0][0].shape[1], dataset[0][0].shape[2])

    # Initialize GEEX explainer
    num_masks, sigma = 5000, 1.0
    expl_ge = geex.GEEX(num_masks, sigma, input_shape)

    output_dir = 'explained_images_geex'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over uploaded images
    for i, (images, labels) in enumerate(dataloader):
        images = images.numpy()  # shape: (1, C, H, W)
        images_tensor = torch.tensor(images).to(device_type)

        # Assuming expl_ge.explain(model, images_tensor) returns a relevance map list
        # If the method or return type differ, adjust accordingly
        rel_map = expl_ge.explain(model, images_tensor)[0]

        # Save relevance map as a heatmap
        cmap = 'jet'
        output_filename_geex = f'{output_dir}/explanation_geex_{i}.png'
        plt.imsave(output_filename_geex, rel_map, cmap=cmap)
        print(f'Saved GEEX explanation: {output_filename_geex}')

def calculate_spade(nb_classes, model, x_test, NLP):
    """计算 SPADE 分数的函数。
    假设在此处选择一部分数据来提取输入输出特征。
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # 转换为张量
    #x_test = x_test.to(device)

    if NLP:
        text_dataset = TextDataset(x_test[:100])
        dataloader = DataLoader(text_dataset, batch_size=32, shuffle=False)

        input_features = []
        output_features = []

        print("提取NLP输入和输出特征...")
        with torch.no_grad():
            for input_ids, attention_mask in dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # 输入特征：词嵌入平均值
                embeddings = model.embeddings(input_ids)
                avg_embeddings = embeddings.mean(dim=1).cpu().numpy()
                input_features.append(avg_embeddings)

                # 输出特征：最后一层隐藏状态平均值
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                avg_output = last_hidden_state.mean(dim=1).cpu().numpy()
                output_features.append(avg_output)

        input_features = np.concatenate(input_features, axis=0)
        output_features = np.concatenate(output_features, axis=0)
        score = spade_score(input_features, output_features)

    else:
        with torch.no_grad():
            # 输入特征为原始像素展平
            input_features = x_test.view(x_test.size(0), -1).cpu().numpy()
            
            # 输出特征为模型最后一层的输出
            outputs = model(x_test).cpu().numpy()
        score = spade_score(input_features, outputs)

    print(f"SPADE Score: {score}")
    return score

def calculate_spade_single(nb_classes, model, x_test, sample_index=0):
    """
    Evaluate SPADE score for a single data sample.
    To compute SPADE, multiple data points are needed to build a k-NN graph.
    Thus, we select this single data point along with a subset of the test set (e.g., first 100 or 500 samples).
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Ensure sample_index is in range
    if sample_index < 0 or sample_index >= x_test.shape[0]:
        raise ValueError("sample_index is out of range.")

    # Select a subset of data, e.g., first 500 samples including the specified sample_index
    max_samples = min(500, x_test.shape[0])
    if sample_index >= max_samples:
        # Use the first (max_samples - 1) plus this sample_index
        indices = list(range(max_samples - 1))
        indices.append(sample_index)
    else:
        # Directly use the first max_samples
        indices = list(range(max_samples))

    x_sub = x_test[indices]
    x_sub = x_sub.to(device)

    with torch.no_grad():
        input_features = x_sub.view(x_sub.size(0), -1).cpu().numpy()
        outputs = model(x_sub).cpu().numpy()

    score = spade_score(input_features, outputs)
    print(f"Single Sample SPADE Score (focus on sample {sample_index}): {score}")
    return score

if __name__ == '__main__':
    # Read YAML configuration file
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open('config.yaml', 'r') as file:
        config = yaml_loader.load(file)
    load_path = config['model_path']
    trigger_path = config['trigger_path']

    # Define ArgumentParser instance
    parser = argparse.ArgumentParser(
        prog='get_clever_scores',
        description='Calculate CLEVER scores',
    )

    parser.add_argument('-d', '--dataset', required=True, help='Dataset to import (mnist, cifar10, mydata)')
    parser.add_argument('-t', '--task_need', required=True, type=str, help='Task to perform: robustness, privacy, poison, or explain')
    parser.add_argument('-c', '--nb_classes', required=True, type=int, help='Number of classes')
    parser.add_argument('-s', '--patch_size', type=int, help='Patch size for poison data, only for data poisoning task')
    parser.add_argument('-test', '--check_attack_effect', action='store_true', help='Check attack effect, only for data poisoning task')
    parser.add_argument('-ch', '--num_channels', type=int, help='Number of channels in uploaded images, only for model explanation task')
    parser.add_argument('--sample_index', type=int, default=0, help='Sample index for single data SPADE evaluation in robustness_poisonability task')
    parser.add_argument('--NLP', action='store_true', help='Use NLP model for SPADE calculation')
    parser.add_argument('--load_model', action='store_true', help='Use pre-trained')



    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'mnist':
        x_train, y_train, x_test, y_test, min_value, max_value = load_mnist_dataset()
    elif args.dataset == 'cifar10':
        print('Loading CIFAR-10 dataset...')
        x_train, y_train, x_test, y_test, min_value, max_value = load_cifar10_dataset()
    elif args.dataset == 'imdb':
        x_train, y_train, x_test, y_test, min_value, max_value = load_imdb_dataset()    
    elif args.dataset == 'mydata':
        x_train, y_train, x_test, y_test, min_value, max_value = load_mydata()
    else:
        raise ValueError('Please specify a correct dataset: mnist, cifar10, or mydata.')

    # Define input shape
    #input_shape = tuple(x_train.shape[1:])

    # Load model
    if args.NLP:
        #model = BertModel.from_pretrained('bert-base-uncased') #for decoding->spade
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) #for classification->privacy
    elif args.load_model:
        model = Net()
        model.load_state_dict(torch.load(load_path))
    else:
        model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.load_state_dict(torch.load(load_path))
    model.eval()

    # Call the corresponding function based on the provided task
    if args.task_need == 'robustness_clever':
        clever_score = calculate_clever_scores(args.nb_classes, model, x_test)
    elif args.task_need == 'robustness_spade':
        spade_score_val = calculate_spade(args.nb_classes, model, x_test, NLP=args.NLP)
    elif args.task_need == 'privacy':
        calculate_SHAPr(args.nb_classes, model, x_train, y_train, x_test, y_test, NLP=args.NLP)
    elif args.task_need == 'poison':
        if args.patch_size is None:
            raise ValueError('Please provide --patch_size for data poisoning task.')
        poison(args.nb_classes, args.check_attack_effect, trigger_path, args.patch_size, x_train, x_test, y_train, y_test, min_value, max_value, model)
    elif args.task_need == 'explain':
        if args.num_channels is None:
            raise ValueError('Please provide --num_channels for model explanation task.')
        explain(args.nb_classes, args.num_channels, model)
    elif args.task_need == 'explain_geex':
        if args.num_channels is None:
            raise ValueError('Please provide --num_channels for model explanation task.')
        explain_geex(args.nb_classes, args.num_channels, model)
    elif args.task_need == 'robustness_poisonability':
        # SPADE scores were calculated by selecting a specific sample as well as a small data set, reflecting the easy neutrality of that data point
        calculate_spade_single(args.nb_classes, model, x_test, sample_index=args.sample_index)
    else:
        print('Please specify a correct task: robustness, privacy, poison, or explain.')