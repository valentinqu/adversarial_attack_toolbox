# **Adversarial Attack Toolbox**

This repository contains a set of utilities for evaluating and performing adversarial attacks on machine learning models. It includes functionality for calculating CLEVER scores, assessing privacy vulnerabilities, generating poisoned data, and explaining model predictions using LIME.

## **Features**

- **CLEVER Scores Calculation**: Assess the robustness of a model by calculating [CLEVER scores](https://openreview.net/pdf?id=BkUHlMZ0b).  
  A higher CLEVER score indicates better network robustness, as the smallest hostile disturbance may have a larger Lp norm. The value range depends on the radius size, which is 0 - 5 by default, and can be modified in the function `toolbox/compute_untargeted_clever()`.

- **SPADE Scores Calculation**: Assess the robustness of a model by calculating [SPADE scores](https://arxiv.org/pdf/2102.03716).  
  The SPADE score ranges between 0 and 1, where 1 indicates that the model is robust and accurately learns the input-output structure, while values closer to 0 suggest the model might not be capturing the essential data structure.

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

## **Environment Installation Options**

You can set up the environment using either Conda or Docker.

### **Option 1: Using Conda**

To create and activate a Conda environment, follow these steps:

```bash
conda create -n adversarial_env python=3.7
conda activate adversarial_env
pip install -r requirements.txt
```
### **Option 2: Using Docker**
Alternatively, you can use Docker to set up and run the toolbox. To build and run the Docker container, use the following commands:

```bash
docker build -t adversarial-attack-toolbox .
bash run_docker.sh # create docker container
```

## **Configuration**

# If you want to use Custom Dataset: 

**Method 1 - Custom Own Dataset**:
- **CSV Label File (`labels.csv`)**: This file should contain two columns - the first column is the image file name, and the second column is the corresponding label. Example:
   
     ```csv
     image1.png,0
     image2.png,1
     image3.png,0
     ```
- **Image folder**: Ensure that the image folder contains all image files with filenames matching the names in the CSV.

**Method 2: Use Hugging Face Dataset**:
 
specify the dataset name, text field, and label field in the command line:

    ```bash
    $ python toolbox.py -d hf_dataset --hf_dataset <hf_dataset_name> --hf_text_field <text> --hf_label_field <label>
    
    -- <hf_dataset_name>: Name of the Hugging Face dataset (e.g., "imdb").
    -- <text>: Field name for text data in the Hugging Face dataset (default: "text").
    -- <label>: Field name for labels in the Hugging Face dataset (default: "label").
    ```

# If you want to use Custom Model:
**Method 1 - Custom Own Dataset with PyTorch**:

1. **Adjust `config.yaml` file** in the same directory with the following format:

    ```yaml
    model_path: 'path/to/your/model.pth'
    ```

    - `model_path`: Path to your trained model checkpoint.
    - `trigger_path`: Path to the trigger image used for data poisoning.

2. **Define the `Net` class** in `model.py` to represent your neural network architecture.

3. **Store your `.pth` files** in the `models/` directory (e.g., `models/my_model.pth`).

**Method 2 - Use Hugging Face Model**:
    specify the model_class and model name in HuggingFace.

    ```bash
    $ python toolbox.py -m <model_class> --model_name <model_name>: 

    -- <model_class>: class of the Hugging Face model ("BertModel","BertForSequenceClassification","BerBertForTokenClassificationtModel","BertForQuestionAnswering", etc).
    -- <model_name> name of the Hugging Face model("bert-base-uncased", "gpt2", etc)
    ```

# Only for Data Poisoning

Place the trigger image into the `trigger/` directory, add trigger image path to `config.yaml` file

    ```yaml
    trigger_path: 'path/to/trigger/image.png'
    ```
    - `trigger_path`: Path to the trigger image used for data poisoning.

# Only for  Model explanations:

Place images for processing into the `images_upload/class_0/` 





## **Usage**
If you are using Docker to set up the environment, run commands directly within the container's command line. 

After starting the Docker container, use the following commands:
### **1.Evaluate CLEVER Scores for Model Robustness**

To calculate CLEVER scores for your model, run the following command:

```bash
$ python3 toolbox.py -d <dataset> -t robustness_clever -c <nb_classes> -m <model>

- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Replace with the number of classes in your dataset.
- <model>:Specify your model.
```

### **Example for image **

#### data from local, model from local

```bash
$ python3 ....
```
#### data from hugging face, model from local
```bash
$ python3 ....
```
#### data from local, model from hugging face

#### data from hf, model from hf

### **Example for nlp **

#### data local, model local
not possible

#### data from hugging face, model local
not possible

#### data local, model hf
not possible

#### data hf, model hf
not possible
### **2.Evaluate SPADE Scores for Model Robustness**

To evaluate model privacy using SPADE cores, run:
For Image
```bash
$ python3 toolbox.py -d <dataset>  -t robustness_spade -c <nb_classes> -m <model>
- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model.
```
For NLP
```bash
$ python3 toolbox.py -d <dataset>  -t robustness_spade_NLP -c <nb_classes> -m <model> 
- <dataset>: Specify your dataset (e.g., "imdb").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model.
```

### **3.Evaluate Single SPADE Scores for Data Robustness**
```bash
$ python3 toolbox.py -d <dataset>  -t robustness_poisonability -c <nb_classes> -m <model> --sample_index <sample_index>
- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Replace with the number of classes in your dataset.
- <model>:Specify your model.
- <sample_index>Specify the index of single data for robustness poisonability evaluation(e.g., "5").
```

### **4. Assess Privacy (SHAPr Leakage)**

To evaluate model privacy using SHAPr leakage, run:

For Image
```bash
$ python3 toolbox.py -d <dataset>  -t privacy -c <nb_classes> -m <model>
- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model.
```

For NLP
```bash
$ python3 toolbox.py -d <dataset>  -t privacy_NLP -c <nb_classes> -m <model>
- <dataset>: Specify your dataset (e.g., "imdb").
- <nb_classes>: Specify with the number of classes in your dataset.
- <model>:Specify your model (e.g., "BertForSequenceClassification'", "mymodel").
```

### 5. **Perform Data Poisoning**

To generate poisoned data and evaluate the attack effect, execute:

```bash
$ python3 toolbox.py -d <dataset> -m <model> -t poison -c <nb_classes> -s <patch_size> -test
```
If just need posioned data, execute:

```bash
$ python3 toolbox.py -d <dataset> -m <model> -t poison -c <nb_classes> -s <patch_size>
- <dataset>: Specify your dataset (e.g., "imdb").
- <model>:Specify your model (e.g., "BertForSequenceClassification'", "mymodel").
- <nb_classes>: Specify with the number of classes in your dataset.
- <patch_size> Specify the patch size for the poison data.
```

### **6. Explain Model Predictions Using LIME**

To generate explanations for model predictions, use:

```bash
$ python toolbox.py -t explain -m <model> -c <nb_classes> -ch <num_channels>
- <model>:Specify your model (e.g., "BertForSequenceClassification'", "mymodel").
- <nb_classes>: Specify with the number of classes in your dataset.
- <num_channels>: Specify with the number of channels in your uploaded images.
```
### **7. Explain Model Predictions Using GEEX**

To generate explanations with GEEX for model predictions, use:

```bash
$ python toolbox.py -t explain_geex -m <model> -c <nb_classes> -ch <num_channels>
- <model>:Specify your model (e.g., "BertForSequenceClassification'", "mymodel").
- <nb_classes>: Specify with the number of classes in your dataset.
- <num_channels>: Specify with the number of channels in your uploaded images.
```
## **Example**
### **1. Evaluate CLEVER Scores for Model Robustness:**
```bash
$ python3 toolbox.py -d cifar10 -t robustness_clever -c 10 -m mymodel
```

### **2. Evaluate SPADE Scores for Model Robustness:**
Fir Image:
```bash
$ python3 toolbox.py -t robustness_spade -d cifar10 -c 10 -m mymodel
```
FOR NLP:
```bash
$ python3 toolbox.py -t robustness_spade_NLP -d imdb -c 2 -m BertModel
```

### **3. Evaluate Single SPADE Scores for Data Robustness:**
```bash
$ python3 toolbox.py -d cifar10 -t robustness_poisonability -c 10 -m mymodel --sample_index 6
```

### **4. Assess Privacy (SHAPr Leakage):**
For Image:
```bash
$ python3 toolbox.py -d cifar10 -t privacy -c 10 -m mymodel 
```
For NLP
```bash
$ python3 toolbox.py -d imdb -t privacy_NLP -c 10 -m BertForSequenceClassification
```

### **5. Perform Data Poisoning:**
```bash
$ python3 toolbox.py -d cifar10 -m mymodel -t poison -c 10 -s 8 -test
```

### **6. Explain Model Predictions Using LIME:**
```bash
$ python3 toolbox.py -d mnist -m mymodel -t explain -c 10 -ch 1
```

### **7. Explain Model Predictions Using GEEX:**
```bash
$ python3 toolbox.py -d mnist -m mymodel -t explain_geex -c 10 -ch 1 
```

## **NOTES**
- For data poisoning, adjust the patch_size, learning rates, and other parameters as needed.
- When this phrase ' delete this range if you want to use all samples' appears, it means that it is used to reduce code runtime and needs to be commented out when deployed.
- The following error may occur when using conda to install the environment on ‘MacOs’：‘Preparing metadata (pyproject.toml) did not run successfully.’
FIX：Brew install rust 
