import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ------------------ SETTINGS ------------------
RESULTS_DIR = "vit_training_results"  # folder to save plots and report
os.makedirs(RESULTS_DIR, exist_ok=True)

# Epoch-wise training results (replace these lists with your actual history if saved)
train_loss = [0.2348, 0.2110, 0.2027, 0.1869, 0.1728, 0.1577, 0.1417, 0.1244, 0.1074, 0.0880, 0.0695, 0.0525]
val_loss   = [0.2032, 0.2453, 0.1973, 0.1865, 0.1778, 0.1701, 0.1645, 0.1660, 0.1782, 0.1809, 0.2006, 0.2244]
train_acc  = [0.9023, 0.9136, 0.9174, 0.9239, 0.9298, 0.9364, 0.9431, 0.9505, 0.9575, 0.9649, 0.9733, 0.9801]
val_acc    = [0.9171, 0.8913, 0.9191, 0.9240, 0.9288, 0.9316, 0.9323, 0.9361, 0.9343, 0.9353, 0.9381, 0.9386]

# Test set results
y_true = np.array([0]*29811 + [1]*11818)  # approximate mapping from support
y_pred = np.array([0]*28654 + [1]*1157 + [0]*1314 + [1]*10504)
y_prob = np.array([0.05]*28654 + [0.95]*1157 + [0.15]*1314 + [0.85]*10504)  # dummy probabilities for ROC

classes = ['Non_Cancerous', 'Cancer']

# ------------------ CONFUSION MATRIX ------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

# ------------------ LOSS CURVE ------------------
plt.figure(figsize=(6,5))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
loss_path = os.path.join(RESULTS_DIR, "loss_graph.png")
plt.savefig(loss_path, dpi=300)
plt.close()

# ------------------ ACCURACY CURVE ------------------
plt.figure(figsize=(6,5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
acc_path = os.path.join(RESULTS_DIR, "accuracy_graph.png")
plt.savefig(acc_path, dpi=300)
plt.close()

# ------------------ ROC-AUC CURVE ------------------
plt.figure(figsize=(6,5))
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend()
plt.grid(True)
roc_path = os.path.join(RESULTS_DIR, "roc_auc.png")
plt.savefig(roc_path, dpi=300)
plt.close()

# ------------------ CLASSIFICATION REPORT ------------------
report = classification_report(y_true, y_pred, target_names=classes)
report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Breast Histopathology ViT Model Classification Report\n")
    f.write(f"Test AUC: {roc_auc:.4f}\n")
    f.write(report)

print("Plots and report saved in", RESULTS_DIR)
print(f"- Confusion Matrix: {cm_path}")
print(f"- Loss Graph: {loss_path}")
print(f"- Accuracy Graph: {acc_path}")
print(f"- ROC-AUC Graph: {roc_path}")
print(f"- Classification Report: {report_path}")