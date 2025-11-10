# Adversarial Attack Toolbox - GUI User Guide

## Overview

This GUI provides an intuitive interface for the Adversarial Attack Toolbox, allowing you to evaluate machine learning models for robustness, privacy risks, and generate explanations without using command-line interfaces.

## Getting Started

### Prerequisites

- Python environment with required dependencies installed
- Streamlit installed (`pip install streamlit==1.23.1`)
- Access to the adversarial attack toolbox backend

### Running the GUI

```bash
streamlit run streamlit_app.py
```

## User Interface Guide

### Step 1: Select Task Type

Choose between two main analysis types:

- **NLP (Natural Language Processing)**: For text classification and sentiment analysis models
- **Image (Image Processing)**: For computer vision and image classification models

### Step 2: Configuration

#### For NLP Tasks

**Dataset Selection:**

- HuggingFace Dataset

  : Use pre-configured datasets

  - Currently supported: `imdb` (movie reviews sentiment analysis)
  - Future datasets (view-only): rotten_tomatoes, amazon_polarity, yelp_polarity, ag_news, emotion, sst2

- Customised dataset

  : Upload your own data

  - **train.csv**: Required file with `text` and `label` columns
  - **test.csv**: Optional testing data with same format

**Model Selection:** Choose from pre-trained models or custom models:

- `bert-base-uncased`: Google BERT base model
- `roberta-base`: Facebook RoBERTa model
- `distilbert-base-uncased`: Lightweight BERT variant
- `mymodel`: Your custom local model

**Analysis Methods (select multiple):**

- **SPADE**: Robustness analysis (scores 0-1, higher = more robust)
- **SHAPr**: Privacy risk analysis (scores 0-1, higher = more risk)
- **LIME**: Model explanation generation

#### For Image Tasks

**Dataset Selection:**

- **HuggingFace Dataset**: `cifar10` currently supported

- Customised dataset

  : Upload your image data

  - **labels.csv**: CSV with image filenames and corresponding labels
  - **Image files**: PNG, JPG, or JPEG files matching the labels.csv

**Model Selection:**

- Currently only supports `mymodel` (custom local models)

**Analysis Methods (select multiple):**

- **CLEVER**: Robustness evaluation using CLEVER scores
- **SPADE**: Robustness analysis (default selected)
- **SHAPr**: Privacy risk analysis
- **POISON**: Data poisoning attack simulation
- **LIME**: Local interpretable model explanations
- **GEEX**: Advanced gradient-based explanations

**Additional Parameters:**

- **Image channels**: 1 (grayscale) or 3 (color)
- **Patch size**: For poisoning attacks (1-32 pixels)
- **Test attack effectiveness**: Enable to evaluate poisoning success

### Step 3: Execution

**Before Running:**

- Ensure all required parameters are configured
- Check estimated execution time
- Verify dataset and model compatibility

**Execution Options:**

- **Operating analysis**: Start the analysis process
- **Reconfiguration**: Reset and modify settings

**During Execution:**

- Real-time progress indicators show current status
- Live output displays analysis progress
- Individual method completion status is tracked

**Results Display:**

- **Combined Results**: Summary of all completed analyses

- **Method-specific Results**: Detailed scores and interpretations

- Score Interpretations

  :

  - SPADE scores > 0.8: Good robustness
  - SHAPr scores > 0.8: High privacy risk
  - CLEVER scores < 2.0: Good robustness

- **Detailed Logs**: Expandable sections with full output

**Result Actions:**

- **Save All Results**: Export complete analysis to timestamped file
- **Reset Results**: Clear current session and start fresh

### Score Interpretations

**SPADE (Robustness):**

- Range: 0-1
- Higher scores indicate better robustness
- 1.0: Good robustness
- 0.95-1.0: Moderate robustness
- < 0.95: Low robustness

**SHAPr (Privacy Risk):**

- Range: 0-1
- Higher scores indicate higher privacy risk
- 0.8: High risk
- 0.5-0.8: Moderate risk
- < 0.5: Low risk

**CLEVER (Image Robustness):**

- Lower scores indicate better robustness
- < 2.0: Good robustness
- 2.0-4.0: Moderate robustness
- 4.0: Vulnerable to attacks

## Troubleshooting

### Common Issues

**Network Connection Errors:**

- Ensure internet connectivity for HuggingFace model downloads
- Consider using local models if connection is unstable

**File Upload Issues:**

- Check CSV format matches requirements exactly
- Ensure image files match filenames in labels.csv
- Verify file sizes are reasonable

**Execution Failures:**

- Check model compatibility with chosen dataset
- Ensure sufficient system resources
- Review error logs in detailed output sections

**Model Loading Errors:**

- Verify custom models are properly configured
- Check model file paths and dependencies

### Performance Tips

- Start with smaller datasets for testing
- Use local models when possible to avoid download delays
- Monitor system resources during execution
- Save results frequently for long analyses

## Best Practices

1. **Test with Small Datasets First**: Validate your configuration before running large analyses
2. **Check Compatibility**: Ensure your data format matches the expected structure
3. **Monitor Progress**: Use real-time outputs to track analysis progress
4. **Save Results**: Export results after completion for later reference
5. **Review Logs**: Check detailed logs if results seem unexpected

## Support

For technical issues or questions about specific analysis methods, refer to the main project documentation or check the detailed logs provided in the GUI for troubleshooting information.