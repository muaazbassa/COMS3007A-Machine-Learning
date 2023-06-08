import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Rotate Data Function
def rotate_data(data, angle):
    rotated_data = np.concatenate((data[:, angle:], data[:, :angle]), axis=1)
    return rotated_data

# Add Noise Function
def add_noise(data, mean, std_dev):
    noisy_data = data + np.random.normal(mean, std_dev, size=data.shape)
    return noisy_data


# Read the data
df_features = pd.read_csv('traindata.txt', delimiter=',', header=None)
df_labels = pd.read_csv('trainlabels.txt', header=None)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_features,
    df_labels,
    test_size=0.3,
    random_state=42
)

# Convert the data to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Data augmentation - random perturbations
augmented_X_train = []
augmented_y_train = []

for i in range(len(X_train)):
    original_data = X_train[i]
    augmented_X_train.append(original_data)
    augmented_y_train.append(y_train[i])

    # Apply random perturbations
    perturbed_data = original_data + np.random.normal(0, 0.1, size=original_data.shape)
    augmented_X_train.append(perturbed_data)
    augmented_y_train.append(y_train[i])

# Convert augmented data to numpy arrays
augmented_X_train = np.array(augmented_X_train)
augmented_y_train = np.array(augmented_y_train)

# Concatenate augmented data with original data
X_train = np.concatenate([X_train, augmented_X_train], axis=0)
y_train = np.concatenate([y_train, augmented_y_train], axis=0)

# Shuffle the augmented data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Rotate the data
X_train = rotate_data(X_train, angle=10)
X_test = rotate_data(X_test, angle=10)

# Add noise to the data
X_train = add_noise(X_train, 0, 0.1)
X_test = add_noise(X_test, 0, 0.1)

# Reshape the data for CNN input
X_train = X_train.reshape(-1, 71, 1)
X_test = X_test.reshape(-1, 71, 1)

# Define some constants
INPUT_SHAPE = (71, 1)  # Number of input features and channels
NUM_CLASSES = 10  # Number of output classes (0-9)
LEARNING_RATE = 0.025  # Adjust as necessary

# Create an ensemble of CNN models
ensemble_size = 10
ensemble = []

# Train individual CNN models
for _ in range(ensemble_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.BatchNormalization(),  # Add batch normalization
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add batch normalization
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu'),  # Additional hidden layer
        tf.keras.layers.BatchNormalization(),  # Add batch normalization
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),  # Additional hidden layer
        tf.keras.layers.BatchNormalization(),  # Add batch normalization
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Define an Adam optimizer with the desired learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    model.fit(X_train, y_train, epochs=30, batch_size=64)

    # Make predictions on the test set
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    print('Model accuracy: {:.2%}'.format(accuracy))

    ensemble.append(model)

# Save the ensemble model
for i, model in enumerate(ensemble):
    model.save(f'ensemble_model_{i}.h5')

# Make predictions on the test set using the ensemble
y_test_pred_prob = np.zeros((len(y_test), NUM_CLASSES))
for model in ensemble:
    y_test_pred_prob += model.predict(X_test)

y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Calculate accuracy of the ensemble
accuracy = accuracy_score(y_test, y_test_pred)
print('Ensemble accuracy: {:.2%}'.format(accuracy))
