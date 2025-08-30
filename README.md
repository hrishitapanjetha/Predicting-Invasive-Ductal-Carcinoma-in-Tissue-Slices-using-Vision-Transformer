# Predicting Invasive Ductal Carcinoma in Tissue Slices using Vision Transformer

This project fine-tunes a pretrained Vision Transformer (ViT) model (vit_base_patch16_224) on breast histopathology images to classify benign (0) and malignant (1) tissue.
The model is implemented using PyTorch and the timm library, and uses data augmentation, cosine learning rate scheduler with warmup, mixed precision training, and early stopping for optimized training.

The original dataset should be structured by patient folders:
breast_cancer_data/
├── patient_1/
│   ├── 0/
│   └── 1/
├── patient_2/
│   ├── 0/
│   └── 1/
...

The dataset reorganization script (data_script_vit.py) converts the original patient-based folder structure into ImageFolder format for PyTorch.
The output_vit folder contains the results and evaluation metrices.
