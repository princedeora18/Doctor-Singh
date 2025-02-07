import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set paths for datasets
train_data_dir = r'path_to_dataset'
val_data_dir = r'path_to_dataset'
test_data_dir = r'path_to_dataset'

# Parameters
batch_size = 32
img_height = 150
img_width = 150

# Function to create datasets
def create_dataset(directory, subset):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset=subset,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

# Create training and validation datasets
train_ds = create_dataset(train_data_dir, 'training')
val_ds = create_dataset(val_data_dir, 'validation')

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Building the model with explicit input shape
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification: Pneumonia vs Normal
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build model explicitly (fix for summary error)
model.build(input_shape=(None, img_height, img_width, 3))

# Print model summary
model.summary()

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping]
)

# Evaluate the model with test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the trained model
model.save('Updated_pneumonia_detection_model.h5')
