import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths for datasets
train_data_dir = 'Your/Path/train'
val_data_dir = 'Your/Path/test'
test_data_dir = 'Your/Path/val'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Test data only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)  # Validation data only rescaling

# Loading data from directories
train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

# Building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification: Pneumonia vs Normal
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Training the model
model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=10,
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)

# Evaluate the model with test data
test_loss, test_acc = model.evaluate(test_data, steps=test_data.samples // test_data.batch_size)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Save the trained model
model.save('pneumonia_detection_model.h5')
