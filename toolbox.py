import argparse
import ruamel.yaml as yaml
import torch
import transformers  # Import for general utility

# Import Custom Modules
from model import Net  # Ensure Net class is defined in model.py
from mydata import load_mydata  # Ensure load_mydata is implemented in mydata.py
from utils import *  # Import utility functions
from toolbox_funcs import *  # Import additional toolbox functions


def main():
    """
    Main function to handle argument parsing, dataset loading, model loading,
    and executing the specified machine learning task (robustness, privacy, poisoning, or explanation).
    """
    
    # Load YAML configuration file
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open('config.yaml', 'r') as file:
        config = yaml_loader.load(file)

    load_path = config['model_path']  # Model checkpoint path
    trigger_path = config['trigger_path']  # Poisoning attack trigger path
    hf_model_path = config['hf_model_folder_path']  # Hugging Face model path

    # Initialize argument parser
    parser = argparse.ArgumentParser(
        prog='get_clever_scores',
        description='Perform various ML security and interpretability tasks',
    )

    # Dataset selection arguments
    parser.add_argument('-d', '--dataset', required=False, help='Dataset to use (mnist, cifar10, mydata, hf_dataset)')
    parser.add_argument('--hf_dataset', type=str, help='Hugging Face dataset name')
    parser.add_argument('--hf_text_field', type=str, default='text', help='Text field name in the Hugging Face dataset')
    parser.add_argument('--hf_label_field', type=str, default='label', help='Label field name in the Hugging Face dataset')

    # Model selection arguments
    parser.add_argument('-m', '--model', required=True, help='Model type (e.g., BertModel, BertForSequenceClassification, mymodel)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Hugging Face model name (default: bert-base-uncased)')

    # Task selection argument
    parser.add_argument('-t', '--task_need', required=True, type=str, help='Task type: robustness, privacy, poison, or explain')

    # Additional arguments for specific tasks
    parser.add_argument('-c', '--nb_classes', required=True, type=int, help='Number of classes in the dataset')
    # Only for POISIONING task
    parser.add_argument('-s', '--patch_size', type=int, help='Patch size for data poisoning (only for poisoning task)')
    parser.add_argument('-test', '--check_attack_effect', action='store_true', help='Evaluate attack effectiveness (only for poisoning task)')
    # Only for EXPLANATION task
    parser.add_argument('-ch', '--num_channels', type=int, help='Number of image channels (only for explanation task)')
    # Only for ROBUSTNESS_POISONABILITY task
    parser.add_argument('--sample_index', type=int, default=0, help='Sample index for SPADE evaluation (robustness_poisonability task)')

    
    args = parser.parse_args()
    print(args.task_need)
    if args.task_need != 'explain' and args.task_need != 'explain_geex':

        # Load dataset 
        if args.dataset == 'mnist':
            x_train, y_train, x_test, y_test, min_value, max_value = load_mnist_dataset()
        elif args.dataset == 'cifar10':
            x_train, y_train, x_test, y_test, min_value, max_value = load_cifar10_dataset()
        elif args.dataset == 'imdb':
            x_train, y_train, x_test, y_test, min_value, max_value = load_imdb_dataset()
        elif args.dataset == 'mydata':
            x_train, y_train, x_test, y_test, min_value, max_value = load_mydata()
        elif args.dataset == 'hf_dataset':
            if args.hf_text_field is None or args.hf_label_field is None:
                raise ValueError('Please specify both --hf_text_field and --hf_label_field for Hugging Face dataset')
            x_train, y_train, x_test, y_test, min_value, max_value = load_hf_dataset(args.hf_dataset, args.hf_text_field, args.hf_label_field)
        else:
            raise ValueError('Invalid dataset choice! Choose from mnist, cifar10, mydata, hf_dataset.')

    # Load model 
    if args.model == 'mymodel':
        model = Net()
        model.load_state_dict(torch.load(load_path))
        print(f"Successfully loaded model: {args.model}")
    else:
        try:
            model_class = getattr(transformers, args.model, None)
            if model_class:
                model = model_class.from_pretrained(args.model_name)
                print(f"Successfully loaded Hugging Face model: {args.model}.{args.model_name}")
            else:
                print('Model not found. Please specify a custom model.')
        except ImportError:
            print(f"Failed to import {args.model}. Ensure it is installed correctly.")
        except AttributeError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")

    # Execute the selected task
    model.eval()

    if args.task_need == 'robustness_clever':
        calculate_clever_scores(args.nb_classes, model, x_test)
    
    elif args.task_need == 'robustness_spade':
        calculate_spade(args.nb_classes, model, x_test, NLP=False)

    elif args.task_need == 'robustness_spade_NLP':
        calculate_spade(args.nb_classes, model, x_test, NLP=True)

    elif args.task_need == 'privacy':
        calculate_SHAPr(args.nb_classes, model, x_train, y_train, x_test, y_test, NLP=False)

    elif args.task_need == 'privacy_NLP':
        calculate_SHAPr(args.nb_classes, model, x_train, y_train, x_test, y_test, NLP=True)

    elif args.task_need == 'poison':
        if args.patch_size is None:
            raise ValueError('Please specify --patch_size for data poisoning task.')
        poison(args.nb_classes, args.check_attack_effect, trigger_path, args.patch_size, x_train, x_test, y_train, y_test, min_value, max_value, model)

    elif args.task_need == 'explain':
        if args.num_channels is None:
            raise ValueError('Please specify --num_channels for model explanation task.')
        explain(args.nb_classes, args.num_channels, model)

    elif args.task_need == 'explain_geex':
        if args.num_channels is None:
            raise ValueError('Please specify --num_channels for model explanation task.')
        explain_geex(args.nb_classes, args.num_channels, model)

    elif args.task_need == 'robustness_poisonability':
        calculate_spade_single(args.nb_classes, model, x_test, sample_index=args.sample_index)

    else:
        print('Invalid task type! Choose from robustness, privacy, poison, or explain.')


if __name__ == '__main__':
    main()
