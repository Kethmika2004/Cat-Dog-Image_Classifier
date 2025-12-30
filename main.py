# ===============================================================
# 🐱🐶 Cats vs Dogs Image Classification – CNN with Keras
# ===============================================================
# This project uses a Convolutional Neural Network (CNN) to classify
# images of cats and dogs. The model leverages data augmentation,
# Batch Normalization, and Dropout to improve accuracy and prevent
# overfitting. It predicts probabilities and visualizes results.
# ===============================================================

# 1️⃣ Setup and Imports
try:
    %tensorflow_version 2.x  # Only for Colab
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ===============================================================
# 2️⃣ Download and Prepare Dataset
# ===============================================================
!wget -q https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
!unzip -q cats_and_dogs.zip

PATH = 'cats_and_dogs'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# ===============================================================
# 3️⃣ Parameters
# ===============================================================
IMG_HEIGHT = 150
IMG_WIDTH = 150
batch_size = 128
epochs = 50

# ===============================================================
# 4️⃣ Data Generators
# ===============================================================
# Training generator with data augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and Test generators (rescale only)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Flow images from directories
train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

test_data_gen = test_image_generator.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ===============================================================
# 5️⃣ Build CNN Model
# ===============================================================
model = Sequential([
    # 1st Conv Block
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # 2nd Conv Block
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # 3rd Conv Block
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # 4th Conv Block
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================================================
# 6️⃣ Callbacks
# ===============================================================
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ===============================================================
# 7️⃣ Train Model
# ===============================================================
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size,
    callbacks=[early_stop, reduce_lr]
)

# ===============================================================
# 8️⃣ Plot Training History
# ===============================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# ===============================================================
# 9️⃣ Prediction Helper Function
# ===============================================================
def plotImages(images_arr, probabilities):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr)*3))
    for img, prob, ax in zip(images_arr, probabilities, axes):
        ax.imshow(img)
        ax.axis('off')
        label = "Dog" if prob > 0.5 else "Cat"
        ax.set_title(f"{label} ({prob*100:.2f}%)")
    plt.show()

# ===============================================================
# 10️⃣ Predict on Test Images
# ===============================================================
all_test_images = []
all_probabilities = []

for i in range(len(test_data_gen)):
    imgs, _ = test_data_gen[i]
    all_test_images.extend(imgs)
    probs = model.predict(imgs).flatten()
    all_probabilities.extend(probs)

all_test_images = np.array(all_test_images)
all_probabilities = np.array(all_probabilities)

# Visualize first 5 test images
plotImages(all_test_images[:5], all_probabilities[:5])

# ===============================================================
# 11️⃣ Evaluate Model Accuracy on Test Set
# ===============================================================
answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
           0, 0, 0, 0, 0, 0]

correct = sum(round(p) == a for p, a in zip(all_probabilities, answers))
percentage_identified = (correct / len(answers)) * 100

print(f"Your model correctly identified {percentage_identified:.2f}% of the test images.")
if percentage_identified >= 63:
    print("🎉 You passed the challenge!")
else:
    print("⚠️ Keep training or augmenting to improve accuracy!")
