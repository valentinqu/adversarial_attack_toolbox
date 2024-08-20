import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np

from art import metrics
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms

#from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
#from art.utils import load_dataset
from art import metrics
from art.estimators.classification import PyTorchClassifier

from lime import lime_image
# from skimage.segmentation import mark_boundaries


class CleverScoreCalculator:
    def __init__(self, model, criterion, optimizer, input_shape, nb_classes, 
                 #transf, 
                 #x_train=None, 
                 #y_train=None, 
                 #batch_size=4, 
                 #nb_epochs=2, 
                 #train=False, 
                 device_type='cpu'):
        """
        Initialize the CleverScoreCalculator object and optionally train the classifier or just initialize LIME.

        :param model: PyTorch model object.
        :param criterion: Loss function.
        :param optimizer: Optimizer.
        :param input_shape: Shape of the input image.
        :param nb_classes: Number of classes for classification.
        :param transf: Preprocessing and ToTensor() transformation.
        :param x_train: Feature matrix for training data. Can be omitted if Train=True.
        :param y_train: Label vector for training data. Can be omitted if Train=True.
        :param batch_size: Batch size during training, default is 200.
        :param nb_epochs: Number of epochs during training, default is 10.
        :param Train (removed): If True, skip model training and only initialize LIME. Default is False.
        :param device_type: Device type for running the model, can be 'cpu' or 'gpu'.
        """
        # Create ART's PyTorchClassifier
        self.classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),  # Modify min_pixel_value and max_pixel_value as needed
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            device_type=device_type
        )
        '''
        if train:  # Train using the built-in CLEVER function, otherwise skip
            if x_train is None or y_train is None:
                raise ValueError("If just_lime=False, x_train and y_train cannot be None")
            self.classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)
        '''
        # Initialize LIME
        self._device = torch.device("cuda" if torch.cuda.is_available() and device_type == 'gpu' else "cpu")
        self.model = model.to(self._device)  # Move the model to the specified device
        #self.transf = transf  # Save the passed transform


    def compute_untargeted_clever(self, x_sample, nb_batches=50, batch_size=10, radius=5, norm=1):
        """Compute the untargeted CLEVER score."""
        clever_untargeted = metrics.clever_u(self.classifier, x_sample, nb_batches, batch_size, radius, norm)
        return clever_untargeted

    def compute_targeted_clever(self, x_sample, target_class, nb_batches=50, batch_size=10, radius=6, norm=1):
        """Compute the targeted CLEVER score."""
        clever_targeted = metrics.clever_t(
            classifier=self.classifier,
            x=x_sample,
            target_class=target_class,
            nb_batches=nb_batches,
            batch_size=batch_size,
            radius=radius,
            norm=norm
        )
        return clever_targeted

    def predict_class(self, x_sample):
        """
        Get the predicted class from the model.

        :param x_sample: Sample to predict
        :return: Predicted class of the sample
        """
        predicted = self.classifier.predict(x_sample)
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
    

if __name__ == "__main__":
    pass
