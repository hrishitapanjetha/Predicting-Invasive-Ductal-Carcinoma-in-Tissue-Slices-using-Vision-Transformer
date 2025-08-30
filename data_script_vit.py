import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
input_dir = "breast_cancer_data"
output_dir = "BreastHistopathologyData"  # New ImageFolder format

# Create output folders
for split in ["train", "val", "test"]:
    for cls in ["0", "1"]:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Collect all images with labels
images, labels = [], []
for patient in os.listdir(input_dir):
    patient_path = os.path.join(input_dir, patient)
    if not os.path.isdir(patient_path):
        continue
    for cls in ["0", "1"]:
        class_path = os.path.join(patient_path, cls)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    images.append(os.path.join(class_path, img_file))
                    labels.append(cls)

# Split into train/val/test (70/15/15)
train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(temp_imgs, temp_lbls, test_size=0.5, stratify=temp_lbls, random_state=42)

# Helper to copy files
def copy_files(imgs, lbls, split):
    for img, lbl in zip(imgs, lbls):
        dst = os.path.join(output_dir, split, lbl, os.path.basename(img))
        shutil.copy(img, dst)

# Copy files
copy_files(train_imgs, train_lbls, "train")
copy_files(val_imgs, val_lbls, "val")
copy_files(test_imgs, test_lbls, "test")

print("Dataset reorganized into ImageFolder format at:", output_dir)