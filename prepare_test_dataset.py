import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# CONFIG (عدلهم لو لزم)
# =========================
DATASET_ROOT = r"C:\Users\tonyh\Desktop\plantvillage dataset\color"
OUTPUT_TEST_ROOT = r"C:\Users\tonyh\Documents\GitHub\Plant-Diseases-Full-Project\test dataset"

SEED = 42
TEST_SIZE = 0.15   # 15% test
VAL_SIZE = 0.15    # 15% val  (هيتعمل split منطقي فقط، بس هننسخ test بس)
TRAIN_SIZE = 0.70  # 70% train

# لو عايز تختار 5 كلاسات بعينهم حطهم هنا بالظبط زي اسماء الفولدرات
# لو سيبتها None => هياخد أول 5 فولدرات أبجديًا
SELECTED_CLASSES = None


def list_classes(dataset_root: str):
    classes = [d.name for d in Path(dataset_root).iterdir() if d.is_dir()]
    classes = sorted(classes)
    return classes


def collect_paths_and_labels(dataset_root: str, class_names: list[str]):
    image_paths = []
    image_labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = Path(dataset_root) / cls
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_paths.append(str(p))
                image_labels.append(idx)
    return image_paths, image_labels


def safe_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def main():
    all_classes = list_classes(DATASET_ROOT)

    if SELECTED_CLASSES is None:
        selected = all_classes[:5]
    else:
        selected = SELECTED_CLASSES

    print("Selected classes (5):")
    for c in selected:
        print(" -", c)

    image_paths, image_labels = collect_paths_and_labels(DATASET_ROOT, selected)
    if len(image_paths) == 0:
        raise RuntimeError("No images found. Check DATASET_ROOT or class folder names.")

    # 70/15/15 split: first split train vs temp(=val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, image_labels,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=SEED,
        stratify=image_labels
    )

    # split temp into val and test equally (15/15)
    val_ratio_of_temp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)  # 0.5
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio_of_temp),
        random_state=SEED,
        stratify=temp_labels
    )

    print(f"\nCounts: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    # Copy ONLY test split into output folder:
    # OUTPUT_TEST_ROOT / class_name / files...
    for src, lab in zip(test_paths, test_labels):
        cls_name = selected[lab]
        fname = os.path.basename(src)
        dst = os.path.join(OUTPUT_TEST_ROOT, cls_name, fname)
        safe_copy(src, dst)

    # Save labels order for GUI
    labels_txt = os.path.join(OUTPUT_TEST_ROOT, "labels.txt")
    with open(labels_txt, "w", encoding="utf-8") as f:
        for cls in selected:
            f.write(cls + "\n")

    print("\n✅ Done. Test dataset created at:")
    print(OUTPUT_TEST_ROOT)
    print("labels.txt saved with class order.")


if __name__ == "__main__":
    main()
