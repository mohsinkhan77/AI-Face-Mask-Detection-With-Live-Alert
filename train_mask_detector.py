# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)

from tensorflow.keras.layers import (
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense,
    Input
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==========================================
# DEFINE DATASET PATHS
# ==========================================

images_path = "dataset/images"
annotations_path = "dataset/annotations"

# ==========================================
# INITIALIZE DATA STORAGE
# ==========================================

data = []
labels = []

# ==========================================
# LOOP THROUGH XML FILES
# ==========================================

for xml_file in os.listdir(annotations_path):

    # Process only XML files
    if xml_file.endswith(".xml"):

        # Full XML path
        xml_path = os.path.join(annotations_path, xml_file)

        # Parse XML file
        tree = ET.parse(xml_path)

        # Get root element
        root = tree.getroot()

        # Image filename
        filename = root.find("filename").text

        # Full image path
        image_path = os.path.join(images_path, filename)

        # Read image
        image = cv2.imread(image_path)

        # Skip invalid images
        if image is None:
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ==========================================
        # LOOP THROUGH OBJECTS
        # ==========================================

        for obj in root.findall("object"):

            # Label
            label = obj.find("name").text

            # Ignore incorrect labels
            if label not in ["with_mask", "without_mask"]:
                continue

            # Bounding box
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Crop face
            face = image[ymin:ymax, xmin:xmax]

            # Skip empty faces
            if face.size == 0:
                continue

            # Resize face
            face = cv2.resize(face, (224, 224))

            # Convert to array
            face = img_to_array(face)

            # Normalize image
            face = preprocess_input(face)

            # Store face
            data.append(face)

            # Store label
            labels.append(label)

# ==========================================
# CONVERT TO NUMPY ARRAYS
# ==========================================

data = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"Total faces loaded: {len(data)}")

# ==========================================
# LABEL ENCODING
# ==========================================

lb = LabelBinarizer()

labels = lb.fit_transform(labels)

labels = to_categorical(labels)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

# ==========================================
# DATA AUGMENTATION
# ==========================================

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ==========================================
# LOAD MOBILENETV2
# ==========================================

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# ==========================================
# BUILD CUSTOM HEAD
# ==========================================

headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(7, 7))(headModel)

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(128, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(2, activation="softmax")(headModel)

# ==========================================
# CREATE FINAL MODEL
# ==========================================

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze pretrained layers
for layer in baseModel.layers:
    layer.trainable = False

# ==========================================
# COMPILE MODEL
# ==========================================

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=INIT_LR),
    metrics=["accuracy"]
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("[INFO] Training network...")

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
   steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# ==========================================
# EVALUATE MODEL
# ==========================================

print("[INFO] Evaluating network...")

#predIdxs = model.predict(testX, batch_size=BS)

#predIdxs = np.argmax(predIdxs, axis=1)

#print(classification_report(
#    testY.argmax(axis=1),
#    predIdxs,
#    target_names=lb.classes_
#))

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model.save("models/mask_detector.keras")

print("Model saved successfully!")