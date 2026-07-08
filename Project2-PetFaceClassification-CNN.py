# Importing TensorFlow and required modules for building CNN
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# Loading dataset from directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150,150),
    batch_size=32
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150,150),
    batch_size=32
)
# Normalizing pixel values between 0 and 1 for better training performance
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# Creating a simple Convolutional Neural Network (CNN)
# CNN is used because it is very effective for image classification tasks
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # Output layer with sigmoid activation for binary classification (cat/dog)
    layers.Dense(1, activation='sigmoid')
])
# Compiling model with optimizer, loss function, and accuracy metric
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Training the model on training dataset and validating on validation dataset
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
# Plotting training and validation accuracy to check performance
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.show()