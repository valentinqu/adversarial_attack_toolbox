# **Adversarial Attack Toolbox**

This repository contains a set of utilities for evaluating and performing adversarial attacks on machine learning models. It includes functionality for calculating CLEVER scores, assessing privacy vulnerabilities, generating poisoned data, and explaining model predictions using LIME.

## **Features**

- **[CLEVER Scores Calculation](https://openreview.net/pdf?id=BkUHlMZ0b)**: Assess the robustness of a model by calculating CLEVER scores.  
  A higher CLEVER score indicates better network robustness, as the smallest hostile disturbance may have a larger Lp norm. The value range depends on the radius size, which is 0 - 5 by default, and can be modified in the function `utils/compute_untargeted_clever()`.

- **Privacy Assessment**: Evaluate model privacy using [SHAPr leakage metrics](https://arxiv.org/abs/2112.02230).  
  A higher final SHAPr score for a training sample means it is more vulnerable to privacy attacks. The values range from 0 - 1.

- **Data Poisoning**: Perform data poisoning attacks to evaluate model resilience against adversarial examples.  
  The [Hidden Trigger Backdoor Attack Sleeper Agent](https://arxiv.org/pdf/2106.08970) is used for this. The default `class_source` is 0 (source class), and `class_target` is 1 (target class for misclassification). These values can be modified in `toolbox.py/class_source` and `toolbox.py/class_target`.

- **Model Explanation**: 
  - **[LIME](https://github.com/marcotcr/lime)**:  
    Create local interpretable explanations using linear surrogate models.  
    - LIME works by perturbing the input and observing the changes in the output, approximating the decision boundaries around the instance.

  - **[GEEX](https://github.com/caiy0220/GEEX)**:  
    Provide high-precision, gradient-like explanations under a black-box setting using path integral-based gradient estimation.  
    - GEEX is particularly effective for image-based tasks and ensures compliance with key attribution properties like **Sensitivity** and **Completeness**, making it a robust alternative for high-dimensional inputs.


## **Requirements**

Ensure you have Python 3.7 installed, along with the following packages:

- `torch 1.13.1`
- `torchvision 0.14.1`
- `lime 0.2.0.1`

## **Configuration**

1. **Create a `config.yaml` file** in the same directory with the following format:

    ```yaml
    model_path: 'path/to/your/model.pth'
    trigger_path: 'path/to/trigger/image.png'
    ```

    - `model_path`: Path to your trained model checkpoint.
    - `trigger_path`: Path to the trigger image used for data poisoning.

2. **Define the `Net` class** in `model.py` to represent your neural network architecture.

3. **Store your `.pth` files** in the `models/` directory (e.g., `models/my_model.pth`).

4. **For Model explanations**: Place images for processing into the `images_upload/class_0/` directory.

5. **For data poisoning**: Place the trigger image into the `trigger/` directory.

6. **Custom Dataset**:
   - **CSV Label File (`labels.csv`)**: This file should contain two columns - the first column is the image file name, and the second column is the corresponding label. Example:
   
     ```csv
     image1.png,0
     image2.png,1
     image3.png,0
     ```
   - **Image folder**: Ensure that the image folder contains all image files with filenames matching the names in the CSV.

## **Environment Installation Options**

You can set up the environment using either Conda or Docker.

### **Option 1: Using Conda**

To create and activate a Conda environment, follow these steps:

```bash
conda create -n adversarial_env python=3.7
conda activate adversarial_env
pip install requirements.txt
```
### **Option 2: Using Docker**
Alternatively, you can use Docker to set up and run the toolbox. To build and run the Docker container, use the following commands:

```bash
docker build -t adversarial-attack-toolbox .
bash run_docker.sh # create docker container
```
## **Usage**
If you are using Docker to set up the environment, run commands directly within the container's command line. 

After starting the Docker container, use the following commands:
### **1.Evaluate CLEVER Scores for Model Robustness**

To calculate CLEVER scores for your model, run the following command:

```bash
$ python3 toolbox.py -d <dataset> -t robustness_clever -c <nb_classes> -m <model>

- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Replace with the number of classes in your dataset.
- <model>:Specify your model (e.g., "ResNet", "mymodel").
```
### **2.Evaluate SPADE Scores for Model Robustness**

To evaluate model privacy using SHAPr leakage, run:
For Image
```bash
$ python3 toolbox.py -d <dataset>  -t robustness_spade -c <nb_classes> -m <model>
- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model (e.g., "ResNet", "mymodel").
```
For NLP
```bash
$ python3 toolbox.py -d <dataset>  -t robustness_spade -c <nb_classes> -m <model> --NLP
- <dataset>: Specify your dataset (e.g., "imdb").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model (e.g., "BertModel'", "mymodel").
```

### **3.Evaluate Single SPADE Scores for Data Robustness**
```bash
$ python3 toolbox.py -d <dataset>  -t robustness_poisonability -c <nb_classes> -m <model> --sample_index <sample_index>
- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Replace with the number of classes in your dataset.
- <model>:Specify your model (e.g., "ResNet", "mymodel").
- <sample_index>Specify the index of single data for robustness poisonability evaluation(e.g., "5").
```

### **4. Assess Privacy (SHAPr Leakage)**

To evaluate model privacy using SHAPr leakage, run:

For Image
```bash
$ python3 toolbox.py -d <dataset>  -t privacy -c <nb_classes> -m <model>
- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model (e.g., "ResNet", "mymodel").
```

For NLP
```bash
$ python3 toolbox.py -d <dataset>  -t privacy -c <nb_classes> -m <model> --NLP
- <dataset>: Specify your dataset (e.g., "imdb").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model (e.g., "BertForSequenceClassification'", "mymodel").
```

### 5. **Perform Data Poisoning**

To generate poisoned data and evaluate the attack effect, execute:

```bash
$ python3 toolbox.py -d <dataset> -t poison -c <nb_classes> -s <patch_size> -test
```
If just need posioned data, execute:

```bash
$ python3 toolbox.py -d <dataset> -t poison -c <nb_classes> -s <patch_size>
- <dataset>: Specify your dataset (e.g., "imdb").
- <nb_classes>: Specify with the number of classes in your dataset.
- <patch_size> Specify the patch size for the poison data.
```

### **6. Explain Model Predictions Using LIME**

To generate explanations for model predictions, use:

```bash
$ python toolbox.py -t explain -c <nb_classes> -ch <num_channels>
- <nb_classes>: Specify with the number of classes in your dataset.
- <num_channels>: Specify with the number of channels in your uploaded images.
```
### **7. Explain Model Predictions Using GEEX**

To generate explanations with GEEX for model predictions, use:

```bash
$ python toolbox.py -t explain_geex -c <nb_classes> -ch <num_channels>
- <nb_classes>: Specify with the number of classes in your dataset.
- <num_channels>: Specify with the number of channels in your uploaded images.
```
## **Example**
### **1. Evaluate CLEVER Scores for Model Robustness:**
```bash
$ python3 toolbox.py -d cifar10 -t robustness_clever -c 10 -m ResNet
```

### **2. Evaluate SPADE Scores for Model Robustness:**
Fir Image:
```bash
$ python3 toolbox.py -t robustness_spade -d cifar10 -c 10 -m ResNet
```
FOR NLP:
```bash
$ python3 toolbox.py -t robustness_spade -d imdb -c 2 -m BertModel --NLP
```

### **3. Evaluate Single SPADE Scores for Data Robustness:**
```bash
$ python3 toolbox.py -d cifar10 -t robustness_poisonability -c 10 -m ResNet --sample_index 6
```

### **4. Assess Privacy (SHAPr Leakage):**
For Image:
```bash
$ python3 toolbox.py -d cifar10 -t privacy -c 10 -m ResNet 
```
For NLP
```bash
$ python3 toolbox.py -d imdb -t privacy -c 10 -m BertForSequenceClassification --NLP
```

### **5. Perform Data Poisoning:**
```bash
$ python3 toolbox.py -d cifar10 -t poison -c 10 -s 8 -test
```

### **6. Explain Model Predictions Using LIME:**
```bash
$ python3 toolbox.py -d mnist -t explain -c 10 -ch 1
```

### **7. Explain Model Predictions Using GEEX:**
```bash
$ python3 toolbox.py -d mnist -t explain_geex -c 10 -ch 1 
```

## **NOTES**
- For data poisoning, adjust the patch_size, learning rates, and other parameters as needed.