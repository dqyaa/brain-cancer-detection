import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your training and testing data directories
train_data_dir = 'C:/Users/USER/Documents/Coding/brain cancer detection/MRI images/Training'
test_data_dir = 'C:/Users/USER/Documents/Coding/brain cancer detection/MRI images/Testing'

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling for testing images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define batch size and target image size
batch_size = 32
target_size = (150, 150)  # Adjust based on your model input size

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and rescale testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Print class indices and labels
print("Class indices:", train_generator.class_indices)
print("Class labels:", train_generator.class_indices.keys())
