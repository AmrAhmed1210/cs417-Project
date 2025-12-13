import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

TRAIN_DIR = "data/New Plant Diseases Dataset(Augmented)/train"
VAL_DIR = "data/New Plant Diseases Dataset(Augmented)/valid"
TEST_DIR = "data/New Plant Diseases Dataset(Augmented)/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

class_labels = os.listdir(TRAIN_DIR)
print("========== Problem Definition ==========")
print(f"Number of classes: {len(class_labels)}")
print(f"Classes: {class_labels}")
print("Image type: RGB")
print("Label type: Single-label")
print("---------------------------------------\n")

def show_sample_images(train_dir, num_samples=5):
    plt.figure(figsize=(15, len(class_labels)*3))
    for i, cls in enumerate(class_labels):
        cls_path = os.path.join(train_dir, cls)
        images = os.listdir(cls_path)
        selected_images = random.sample(images, min(num_samples, len(images)))
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            plt.subplot(len(class_labels), num_samples, i*num_samples + j + 1)
            plt.imshow(img)
            plt.axis('off')
            if j == 0:
                plt.ylabel(cls, fontsize=12)
    plt.suptitle("Sample Images per Class", fontsize=16)
    plt.show()

def plot_class_distribution(train_dir):
    counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in class_labels]
    plt.figure(figsize=(12,6))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Number of Samples per Class")
    plt.ylabel("Count")
    plt.show()
    for cls, count in zip(class_labels, counts):
        print(f"Class '{cls}' has {count} images.")

show_sample_images(TRAIN_DIR)
plot_class_distribution(TRAIN_DIR)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced

def custom_preprocess(img):
    img = img.astype(np.uint8)
    img = apply_clahe(img)
    return img.astype(np.float32)

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    rescale=1./255
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
    color_mode='rgb',
    interpolation='bilinear'
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb',
    interpolation='bilinear'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
    color_mode='rgb',
    interpolation='bilinear'
)

train_classes = train_generator.classes
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)
class_weights = dict(zip(np.arange(len(class_labels)), weights))
print("\nClass weights:", class_weights)

X, y = train_generator[0]
print("\nBatch X shape:", X.shape)
print("Batch y shape:", y.shape)
