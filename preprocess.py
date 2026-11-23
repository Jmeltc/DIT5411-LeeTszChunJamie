import os
import shutil
from collections import defaultdict
import math

SOURCE_DIR = "data_raw"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Create output directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

files = os.listdir(SOURCE_DIR)

print("Total files:", len(files))

# Group images by character
characters = defaultdict(list)
count = 0
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(
        (".png", ".PNG")
    ):
        count += 1

for filename in os.listdir(SOURCE_DIR):

    if filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    ):

        try:
            word = filename.split("_")[0]
            characters[word].append(filename)

        except Exception:
            print(f"Skipped: {filename}")


total_classes = 0
total_train = 0
total_test = 0


for word, files in characters.items():

    # Sort according to image number
    files.sort(
        key=lambda x: int(
            os.path.splitext(x)[0].split("_")[1]
        )
    )

    total_images = len(files)


    train_count = math.floor(total_images * 0.8)


    train_count = min(train_count, 40)


    if train_count >= total_images:
        train_count = total_images - 1

    if train_count <= 0:
        continue

    test_count = total_images - train_count


    train_class_dir = os.path.join(TRAIN_DIR, word)
    test_class_dir = os.path.join(TEST_DIR, word)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    train_files = files[:train_count]
    test_files = files[train_count:]

    # Copy training images
    for file in train_files:

        src = os.path.join(SOURCE_DIR, file)
        dst = os.path.join(train_class_dir, file)

        shutil.copy2(src, dst)

    # Copy testing images
    for file in test_files:

        src = os.path.join(SOURCE_DIR, file)
        dst = os.path.join(test_class_dir, file)

        shutil.copy2(src, dst)

    total_classes += 1
    total_train += len(train_files)
    total_test += len(test_files)

    print(
        f"{word}: "
        f"Total={total_images}, "
        f"Train={len(train_files)}, "
        f"Test={len(test_files)}"
    )

print("\n========== SUMMARY ==========")
print(f"Total image files: {count}")
print(f"Classes : {total_classes}")
print(f"Train   : {total_train}")
print(f"Test    : {total_test}")