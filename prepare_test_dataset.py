"""
prepare_test_dataset.py
Prepares test dataset with exact 70/15/15 split matching the Colab notebook
Saves ONLY the test split (15%) to local directory for the GUI app
"""

import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIGURATION - MATCH NOTEBOOK EXACTLY
# ==========================================================
SEED = 42
np.random.seed(SEED)

# Source dataset path - where your original dataset is located
DATASET_PATH = r"C:\Users\tonyh\Desktop\plantvillage dataset\color"

# Output directory - where TEST SPLIT will be saved (for GUI app)
OUTPUT_DIR = Path(r"C:\Users\tonyh\Documents\GitHub\Plant-Diseases-Full-Project\test dataset")

# Classes - will be sorted alphabetically (matching notebook logic)
ALLOWED_CLASSES = [
    "Tomato___Target_Spot",
    "Pepper,_bell___Bacterial_spot",
    "Grape___Black_rot",
    "Corn_(maize)___Common_rust_",
    "Cherry_(including_sour)___Powdery_mildew"
]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ==========================================================
# FUNCTIONS
# ==========================================================

def collect_image_paths(dataset_path, selected_classes):
    """
    Collect all image paths and labels from dataset.
    Exactly matches notebook: for label_idx, class_name in enumerate(selected_classes)
    """
    image_paths = []
    image_labels = []

    print("\nCollecting images from dataset...")
    for label_idx, class_name in enumerate(selected_classes):
        class_dir = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_dir):
            print(f"⚠ Warning: Class directory not found: {class_dir}")
            continue

        # Exactly matching notebook: files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_ext)]
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]

        print(f"  Class {label_idx} - {class_name}: {len(files)} images")

        for f in files:
            image_paths.append(os.path.join(class_dir, f))
            image_labels.append(label_idx)

    return np.array(image_paths), np.array(image_labels)


def split_dataset_notebook_style(image_paths, image_labels, seed=42):
    """
    Split dataset EXACTLY as done in the notebook:

    From notebook cell:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, image_labels,
        test_size=0.30,
        random_state=SEED,
        stratify=image_labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )

    Result: 70% train, 15% val, 15% test
    """
    print("\n" + "="*70)
    print("Splitting dataset (70/15/15) - EXACTLY AS NOTEBOOK")
    print("="*70)

    # FIRST SPLIT: 70% train, 30% temp (stratified)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, image_labels,
        test_size=0.30,
        random_state=seed,
        stratify=image_labels
    )

    # SECOND SPLIT: 50/50 split of temp into val and test (stratified)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    # Calculate percentages
    total = len(image_paths)
    train_pct = (len(train_paths) / total) * 100
    val_pct = (len(val_paths) / total) * 100
    test_pct = (len(test_paths) / total) * 100

    print(f"\nTrain: {len(train_paths)} images ({train_pct:.1f}%)")
    print(f"Val:   {len(val_paths)} images ({val_pct:.1f}%)")
    print(f"Test:  {len(test_paths)} images ({test_pct:.1f}%)")
    print(f"Total: {total} images")

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def copy_test_files(test_paths, test_labels, class_names, output_dir):
    """
    Copy ONLY test split files to output directory.
    Structure: output_dir/class_name/image.jpg
    (No train/val/test subfolders - just classes directly)
    """
    print("\n" + "="*70)
    print(f"Copying {len(test_paths)} test images to: {output_dir}")
    print("="*70)

    # Clear output directory if it exists
    if output_dir.exists():
        print("\nClearing existing output directory...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy files organized by class
    for i, (path, label) in enumerate(zip(test_paths, test_labels)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(test_paths)} files...")

        class_name = class_names[label]
        dest_class_dir = output_dir / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        src_file = Path(path)
        dest_file = dest_class_dir / src_file.name

        # Copy file
        shutil.copy2(src_file, dest_file)

    print(f"  ✓ Finished copying all {len(test_paths)} test images!")


def save_labels_file(class_names, output_dir):
    """
    Save class names to labels.txt in exact order.
    This file is used by the GUI app to map predictions to class names.
    """
    labels_path = output_dir / "labels.txt"
    with open(labels_path, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print(f"\n✓ Saved class labels to: {labels_path}")


def print_detailed_summary(splits, class_names):
    """Print detailed summary showing per-class distribution"""
    print("\n" + "="*70)
    print("DETAILED SPLIT SUMMARY")
    print("="*70)

    for split_name, (paths, labels) in splits.items():
        print(f"\n{split_name.upper()} SET: {len(paths)} images")
        print("-" * 70)

        # Count images per class
        for idx, class_name in enumerate(class_names):
            count = np.sum(labels == idx)
            percentage = (count / len(labels)) * 100
            print(f"  {idx}. {class_name:45s} : {count:4d} ({percentage:5.1f}%)")

    total = sum(len(paths) for paths, _ in splits.values())
    print("\n" + "="*70)
    print(f"TOTAL IMAGES: {total}")
    print("="*70)


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("="*70)
    print("PREPARING TEST DATASET - MATCHING COLAB NOTEBOOK")
    print("="*70)
    print("\nThis script will:")
    print("  1. Split data 70/15/15 (train/val/test) using notebook logic")
    print("  2. Copy ONLY the test split (15%) to output directory")
    print("  3. Save labels.txt for the GUI app")

    # Verify source dataset path exists
    if not os.path.isdir(DATASET_PATH):
        print(f"\n❌ ERROR: Dataset path not found!")
        print(f"   Path: {DATASET_PATH}")
        print("\n   Please update DATASET_PATH in the script.")
        return

    print(f"\n📁 Source dataset: {DATASET_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")

    # Get and sort classes (matching notebook: selected_classes = sorted(selected_classes))
    selected_classes = [c for c in ALLOWED_CLASSES
                       if os.path.isdir(os.path.join(DATASET_PATH, c))]
    selected_classes = sorted(selected_classes)

    print(f"\n📋 Selected Classes ({len(selected_classes)}):")
    for i, c in enumerate(selected_classes):
        print(f"   {i}. {c}")

    if len(selected_classes) == 0:
        print("\n❌ ERROR: No valid class directories found!")
        print("   Make sure the class folder names match ALLOWED_CLASSES")
        return

    # Collect all image paths and labels (matching notebook logic)
    image_paths, image_labels = collect_image_paths(DATASET_PATH, selected_classes)

    if len(image_paths) == 0:
        print("\n❌ ERROR: No images found in dataset!")
        return

    print(f"\n✓ Total images found: {len(image_paths)}")

    # Split dataset using EXACT notebook logic
    splits = split_dataset_notebook_style(image_paths, image_labels, seed=SEED)

    # Print detailed summary
    print_detailed_summary(splits, selected_classes)

    # Get test split
    test_paths, test_labels = splits['test']

    # Confirm before copying
    print("\n" + "="*70)
    print(f"Ready to copy {len(test_paths)} test images to:")
    print(f"  {OUTPUT_DIR}")
    print("\nThis will create the following structure:")
    print(f"  {OUTPUT_DIR}/")
    for cls in selected_classes:
        print(f"    ├── {cls}/")
    print(f"    └── labels.txt")
    print("="*70)

    response = input("\nProceed with copying? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n❌ Operation cancelled.")
        return

    # Copy test files to output directory
    copy_test_files(test_paths, test_labels, selected_classes, OUTPUT_DIR)

    # Save labels file
    save_labels_file(selected_classes, OUTPUT_DIR)

    # Final summary
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Test images saved: {len(test_paths)}")
    print(f"   • Number of classes: {len(selected_classes)}")
    print(f"   • Output location: {OUTPUT_DIR}")
    print(f"   • labels.txt created: ✓")

    print(f"\n📁 Directory structure created:")
    print(f"   {OUTPUT_DIR}/")
    for cls in selected_classes:
        class_dir = OUTPUT_DIR / cls
        count = len(list(class_dir.glob("*")))
        print(f"     ├── {cls}/ ({count} images)")
    print(f"     └── labels.txt")

    print("\n🎮 You can now run app.py to test your models!")
    print("="*70)


if __name__ == "__main__":
    main()