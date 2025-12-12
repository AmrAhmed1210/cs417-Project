import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

TRAIN_DIR = "../data/New Plant Diseases Dataset(Augmented)/train"
VAL_DIR = "../data/New Plant Diseases Dataset(Augmented)/valid"
TEST_DIR = "../data/New Plant Diseases Dataset(Augmented)/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

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
    vertical_flip=False,
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
class_labels = list(train_generator.class_indices.keys())

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)

class_weights = dict(zip(np.arange(len(class_labels)), weights))
print("Class weights:", class_weights)

X, y = train_generator[0]
print("Batch X shape:", X.shape)
print("Batch y shape:", y.shape)
print("Classes:", class_labels)