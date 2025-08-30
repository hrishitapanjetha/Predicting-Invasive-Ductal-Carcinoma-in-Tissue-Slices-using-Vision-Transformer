"""
Optimized Vision Transformer training script for breast histopathology dataset.
Dataset is expected in ImageFolder format:
BreastHistopathology/
├── train/
│   ├── 0/
│   └── 1/
├── val/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
"""

import os
import math
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

DATA_ROOT = "BreastHistopathologyData"  
OUT_DIR = Path("./vit_training_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 48
NUM_EPOCHS = 20
LR = 3e-5
WEIGHT_DECAY = 1e-2
MODEL_NAME = "vit_base_patch16_224"
PRETRAINED = True
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 4

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def save_txt(path, text):
    with open(path, "w") as f:
        f.write(text)


def load_datasets(data_root):
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=val_transform)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=val_transform)
    return train_ds, val_ds, test_ds


def create_model():
    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
    return model


def get_cosine_scheduler(optimizer, warmup_epochs=3, max_epochs=NUM_EPOCHS):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_and_evaluate(data_root, out_dir=OUT_DIR):
    device = DEVICE
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = load_datasets(data_root)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_scheduler(optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_auc = 0.0
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        # Training
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                val_loss += loss.item() * imgs.size(0)
                val_total += imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_probs)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save({'model_state': model.state_dict(), 'epoch': epoch+1, 'val_auc': val_auc}, out_dir / 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    # Final evaluation on test set
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_auc = roc_auc_score(all_labels, all_probs)
    test_f1 = f1_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Save results
    save_txt(out_dir / 'results_readable.txt', f"Test AUC: {test_auc:.4f}\nTest F1: {test_f1:.4f}\nPrecision: {test_precision:.4f}\nRecall: {test_recall:.4f}\n\n{cls_report}\nConfusion Matrix:\n{cm}")
    print('Training complete. Results saved in', out_dir)

if __name__ == '__main__':
    train_and_evaluate(DATA_ROOT, OUT_DIR)
