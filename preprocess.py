import os
import shutil
from collections import defaultdict
import math
import cv2
import numpy as np
import random

SOURCE_DIR = "data_raw"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Create output directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

files = os.listdir(SOURCE_DIR)

print("Total files:", len(files))

#Data Argumentation
def augment_image(image):

    height, width = image.shape[:2]

    # 1. Rotation (-15° to +15°)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((width / 2, height / 2),angle,1.0)
    image = cv2.warpAffine(image,M,(width, height),borderMode=cv2.BORDER_REPLICATE)

    # 2. Scaling (90% - 110%)
    scale = random.uniform(0.9, 1.1)
    scaled_w = int(width * scale)
    scaled_h = int(height * scale)
    image = cv2.resize(image,(scaled_w, scaled_h))
    image = cv2.resize(image,(width, height))


    # 3. Translation
    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)
    M = np.float32([[1, 0, tx],[0, 1, ty]])
    image = cv2.warpAffine(image,M,(width, height),borderMode=cv2.BORDER_REPLICATE)

    # 4. Contrast Adjustment
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-10, 10)
    image = cv2.convertScaleAbs(image,alpha=alpha,beta=beta)

    # 5. Gaussian Noise
    noise = np.random.normal(0,8,image.shape)
    image = image.astype(np.float32) + noise
    image = np.clip(image,0,255).astype(np.uint8)
    return image

# Group images by character
characters = defaultdict(list)
count = 0
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(
        (".png", ".PNG")
    ):
        count += 1

for filename in os.listdir(SOURCE_DIR):

    if filename.lower().endswith((".png")):
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
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))
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

    # Generate augmented images
    current_train_images = train_files.copy()
    target_size = 200
    augment_count = 0
    while len(current_train_images) < target_size:
        source_file = random.choice(train_files)
        src = os.path.join(SOURCE_DIR, source_file)
        image = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        augmented = augment_image(image)
        new_filename = (f"aug_{augment_count}_"f"{source_file}")
        save_path = os.path.join(train_class_dir,new_filename)
        cv2.imwrite(save_path,augmented)
        current_train_images.append(new_filename)
        augment_count += 1
    
    # Copy testing images
    for file in test_files:
        src = os.path.join(SOURCE_DIR, file)
        dst = os.path.join(test_class_dir, file)
        shutil.copy2(src, dst)

    total_classes += 1
    total_train += len(train_files)
    total_test += len(test_files)

    print(f"{word}: "f"Total={total_images}, "f"Train={len(train_files)}, "f"Test={len(test_files)}")

print("\n========== SUMMARY ==========")
print(f"Total image files: {count}")
print(f"Classes : {total_classes}")
print(f"Train   : {total_train}")
print(f"Test    : {total_test}")
