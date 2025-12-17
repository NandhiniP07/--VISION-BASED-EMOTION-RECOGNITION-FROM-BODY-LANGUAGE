import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration ---
TRAIN_DIR = r'C:\Users\nandh\Documents\train'
TEST_DIR = r'C:\Users\nandh\Documents\test'
IMG_SIZE = (48, 48) # Standard size for many emotion datasets
BATCH_SIZE = 64
NUM_CLASSES = 7

# --- Load Datasets ---
# Keras can automatically load and label data from folders.
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    label_mode='categorical',
    color_mode='grayscale', # Emotions are often better recognized by shape than color
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Print the class names that Keras found
class_names = train_dataset.class_names
print(f"Found class names: {class_names}")

# --- Normalize the data ---
# Create a normalization layer to scale pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# --- Build the CNN Model ---
model = Sequential([
    # Input shape is (48, 48, 1) because images are grayscale
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') # Softmax for multi-class classification
])

model.summary()

# --- Compile and Train the Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks to save the best model and stop early if it's not improving
checkpoint = ModelCheckpoint(
    "emotion_model.h5", 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, # Stop after 10 epochs with no improvement
    restore_best_weights=True
)

print("Starting training...")
history = model.fit(
    train_dataset,
    epochs=50, # Train for more epochs, early stopping will handle the rest
    validation_data=test_dataset,
    callbacks=[checkpoint, early_stopping]
)

print("Training complete! Best model saved as emotion_model.h5")